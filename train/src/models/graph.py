"""可微训练图与动作投影；GRPO可选择原生KV或整段因果回放。"""

import torch
from train.src.models.streaming import replay
from train.src.domain.grpo import action_log_probs



class TrainGraph(torch.nn.Module):
    """DeepSpeed sees the complete model and all trainable media computations."""
    def __init__(self, model, duplex, config):
        super().__init__()
        self.model, self.duplex, self.config = model, duplex, config
        from train.src.models.loading import enable_full_training
        enable_full_training(model)

    def forward(self, record, clip, trajectory=None):
        """SFT返回监督loss；GRPO按配置回放实际轨迹并检查原生动作概率。"""
        from torch.utils.checkpoint import checkpoint
        from train.src.domain.grpo import policy_loss
        config, llm = self.config, self.model.llm
        from train.src.models.cached_replay import checkpoint_vision_embedding
        with checkpoint_vision_embedding(self.model, trajectory is not None
                and config.get('grpo_vision_checkpointing', False)):
            encoded = replay(self.model, self.duplex, record, clip, config, trajectory)
        self.last_stats = dict(sequence_tokens=len(encoded['inputs_embeds']),
                               supervised_tokens=int((encoded['labels'] != -100).sum()))
        if trajectory is None:
            self.last_stats.update({k: encoded[k] for k in ('targets', 'response_count', 'target_chunks', 'tail_silence_seconds')})
        if trajectory is None:
            hidden = llm.model(inputs_embeds=encoded['inputs_embeds'][None], use_cache=False,
                               return_dict=True).last_hidden_state[0]
        elif config.get('grpo_replay_mode', 'native_kv') == 'full_sequence':
            from train.src.models.full_sequence_replay import full_sequence_hidden
            hidden = full_sequence_hidden(llm, encoded['inputs_embeds'])
        else:
            # BF16 full-sequence recomputation can differ materially from native
            # chunked KV decoding. Preserve native feed boundaries and positions
            # for on-policy ratios; retain a differentiable KV cache for backward.
            if config.get('grpo_replay_checkpoint_segments', 0):
                from train.src.models.cached_replay import cached_hidden
                hidden = cached_hidden(llm, encoded['segments'], config['grpo_replay_checkpoint_segments'],
                    checkpoint_feeds=config.get('grpo_replay_checkpoint_feeds', False))
            else:
                past, offset, states = None, 0, []
                for segment in encoded['segments']:
                    pos = torch.arange(offset, offset + len(segment), device=segment.device)[None]
                    out = llm.model(inputs_embeds=segment[None], position_ids=pos,
                        past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    states.append(out.last_hidden_state[0]); offset += len(segment)
                hidden = torch.cat(states)
        if trajectory is None:
            labels, groups = encoded['labels'].to(hidden.device), encoded['groups'].to(hidden.device)
            from train.src.models.sft_loss import grouped_loss
            from train.src.domain.loss_metrics import statistics
            tokenizer = self.duplex.tokenizer
            loss, self.sft_loss_metrics = grouped_loss(llm.lm_head, hidden, labels, groups, config,
                tokenizer.convert_tokens_to_ids('<|listen|>'), tokenizer.convert_tokens_to_ids('<|speak|>'),
                decision_roles=encoded['decision_roles'].to(hidden.device),
                event_roles=encoded['event_roles'].to(hidden.device),
                event_ids=encoded['event_ids'].to(hidden.device))
            self.last_stats.update(statistics(self.sft_loss_metrics))
            self.last_stats.update(statistics(self.sft_loss_metrics, prefix='local_'))
            return loss
        positions = torch.tensor(encoded['positions'], device=hidden.device)
        ids = torch.tensor(encoded['token_ids'], device=hidden.device)
        turns = torch.tensor(encoded['turns'], device=hidden.device)
        if not len(positions):
            raise ValueError('No sampled actions in trajectory')
        log_probs = []
        for start in range(0, len(positions), config['loss_projection_chunk']):
            end = start + config['loss_projection_chunk']
            def project(h, target, turn_open):
                # Native streaming projects one hidden state at a time. A BF16
                # batched GEMM may round large logits differently from the
                # single-token GEMV (0.5 at logits around 74), corrupting PPO
                # ratios even when every replay embedding is identical.
                logits = torch.cat([llm.lm_head(row[None, None, :])[:, -1, :]
                                    for row in h], dim=0)
                law = action_log_probs(logits, self.duplex.decoder.forbidden_token_ids,
                    self.duplex.chunk_eos_token_id, self.duplex.listen_token_id,
                    self.duplex.tts_bos_token_id, turn_open)
                return law.gather(-1, target[:, None]).squeeze(-1)
            log_probs.append(checkpoint(project, hidden[positions[start:end]], ids[start:end], turns[start:end], use_reentrant=False))
        current = torch.cat(log_probs)
        old = torch.tensor([a['old_log_prob'] for c in trajectory['chunks'] for a in c['actions']], device=hidden.device)
        differences = (current.detach() - old).abs()
        error = differences.max()
        worst = int(differences.argmax().item())
        self.last_replay_diagnostics = dict(
            replay_max_logprob_error=error.item(),
            replay_mean_logprob_error=differences.mean().item(),
            replay_p95_logprob_error=torch.quantile(differences.float(), .95).item(),
            replay_worst_action=worst, replay_worst_token_id=int(ids[worst]),
            replay_worst_position=int(positions[worst]),
            replay_worst_old_logprob=float(old[worst]),
            replay_worst_current_logprob=float(current[worst].detach()))
        if 'advantage' not in trajectory and 'local_credit' not in trajectory:
            return current.detach(), error
        local_info = {}
        if 'local_credit' in trajectory:
            from train.src.models.local_policy import local_policy_loss
            loss, ratio, local_info = local_policy_loss(current, old, trajectory, config.get('clip_epsilon', .2),
                advantage_mode=config.get('grpo_advantage_mode', 'local'),
                aggregation=config.get('grpo_loss_aggregation', 'unit_weighted'))
        else:
            loss, ratio = policy_loss(current, old, trajectory['advantage'], config.get('clip_epsilon', .2))
        if config.get('kl_beta', 0):
            from train.src.models.reference_kl import sampled_kl
            if 'reference_log_probs' not in trajectory:
                raise ValueError('Nonzero KL requires frozen-reference action probabilities')
            reference = torch.as_tensor(trajectory['reference_log_probs'], device=current.device, dtype=current.dtype)
            kl = sampled_kl(current, reference, trajectory)
            loss = loss + config['kl_beta'] * kl
            local_info.update(kl_surrogate=kl.detach().item(), kl_contribution=(config['kl_beta'] * kl).detach().item())
        return loss, dict(**self.last_replay_diagnostics, ratio_mean=ratio.mean().item(),
                          actions=len(current), **local_info)
