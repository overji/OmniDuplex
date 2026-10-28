"""可微训练图与监督位置投影。SFT不生成全序列词表logits，GRPO按原生feed边界回放KV。"""

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
        """SFT返回分组加权loss；GRPO返回动作log-prob或策略loss与回放诊断。GRPO保留原生feed分段KV，避免概率分布偏移。"""
        from torch.utils.checkpoint import checkpoint
        from train.src.domain.grpo import policy_loss
        config, llm = self.config, self.model.llm
        encoded = replay(self.model, self.duplex, record, clip, config, trajectory)
        self.last_stats = dict(sequence_tokens=len(encoded['inputs_embeds']),
                               supervised_tokens=int((encoded['labels'] != -100).sum()))
        if trajectory is None:
            self.last_stats.update({k: encoded[k] for k in ('targets', 'response_count', 'target_chunks', 'tail_silence_seconds')})
        if trajectory is None:
            hidden = llm.model(inputs_embeds=encoded['inputs_embeds'][None], use_cache=False,
                               return_dict=True).last_hidden_state[0]
        else:
            # BF16 full-sequence recomputation can differ materially from native
            # chunked KV decoding. Preserve native feed boundaries and positions
            # for on-policy ratios; retain a differentiable KV cache for backward.
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
            loss = hidden.sum() * 0
            active = False
            for group, key in ((1, 'decision_weight'), (2, 'content_weight'), (3, 'call_weight')):
                positions = torch.where((labels[1:] != -100) & (groups[1:] == group))[0]
                if not len(positions) or not config[key]:
                    continue
                active = True
                for ix in positions.split(config['loss_projection_chunk']):
                    def project(h, targets):
                        return torch.nn.functional.cross_entropy(llm.lm_head(h).float(), targets, reduction='sum')
                    loss = loss + config[key] * checkpoint(project, hidden[ix], labels[ix + 1], use_reentrant=False) / len(positions)
            if not active:
                raise ValueError('No supervised tokens with nonzero loss weight')
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
                law = action_log_probs(llm.lm_head(h), self.duplex.decoder.forbidden_token_ids,
                    self.duplex.chunk_eos_token_id, self.duplex.listen_token_id,
                    self.duplex.tts_bos_token_id, turn_open)
                return law.gather(-1, target[:, None]).squeeze(-1)
            log_probs.append(checkpoint(project, hidden[positions[start:end]], ids[start:end], turns[start:end], use_reentrant=False))
        current = torch.cat(log_probs)
        old = torch.tensor([a['old_log_prob'] for c in trajectory['chunks'] for a in c['actions']], device=hidden.device)
        error = (current.detach() - old).abs().max()
        if 'advantage' not in trajectory:
            return current.detach(), error
        loss, ratio = policy_loss(current, old, trajectory['advantage'], config.get('clip_epsilon', .2))
        return loss, dict(replay_max_logprob_error=error.item(), ratio_mean=ratio.mean().item(),
                          actions=len(current))
