"""Frozen-reference scoring and sampled-action KL for streaming GRPO."""
import torch


def sampled_kl(current, reference, trajectory):
    """K3 sampled-action surrogate, balanced across reply and waiting units.

    Samples come from the rollout policy; this is a Monte Carlo surrogate,
    not exact vocabulary KL. Each reply gets one unit irrespective of its
    length; idle/control decisions get one unit per second. Reply and waiting
    categories each receive half the weight when both are present.
    """
    from train.src.models.local_policy import action_groups
    if current.ndim != 1 or current.shape != reference.shape or not current.numel():
        raise ValueError('KL reference action probabilities must align exactly')
    if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
        raise ValueError('Nonfinite KL action probabilities')
    delta = reference.detach().to(device=current.device, dtype=current.dtype) - current
    if bool((delta.detach().abs() > 20).any()):
        raise ValueError('KL log-probability ratio exceeds numerical safety limit')
    values = torch.expm1(delta) - delta
    replies, idle, controls, eligible, count = action_groups(trajectory)
    if count != current.numel() or not eligible:
        raise ValueError('KL action coverage mismatch')
    categories = []
    if replies:
        categories.append(torch.stack([values[positions].mean() for positions in replies.values()]).mean())
    waiting = list(idle.values()) + list(controls.values())
    if waiting:
        categories.append(torch.stack([values[positions].mean() for positions in waiting]).mean())
    if not categories:
        raise ValueError('KL requires sampled action units')
    return torch.stack(categories).mean()


class FrozenReference:
    """Frozen SFT reference, optionally scored on GPU and offloaded between replays."""
    def __init__(self, config, policy_duplex, device=None):
        from train.src.models.loading import load_model
        from train.src.infrastructure.checkpoints import tokenizer_metadata
        self.device = torch.device(device or 'cpu') if config.get('kl_reference_device', 'cpu') == 'cuda_offload' else torch.device('cpu')
        if config.get('kl_reference_device', 'cpu') == 'cuda_offload' and self.device.type != 'cuda':
            raise ValueError('cuda_offload reference requires a CUDA training device')
        backend = config['attention_backend'] if self.device.type == 'cuda' else 'sdpa'
        settings = dict(config, model_path=config['kl_reference_model_path'], attention_backend=backend)
        self.model, self.duplex = load_model(settings, torch.device('cpu'))
        self.model.eval().requires_grad_(False)
        self.model.llm.config.use_cache = True
        if tokenizer_metadata(self.duplex.tokenizer) != tokenizer_metadata(policy_duplex.tokenizer):
            raise ValueError('KL reference and policy tokenizers differ')
        for name in ('chunk_eos_token_id', 'listen_token_id', 'tts_bos_token_id'):
            if getattr(self.duplex, name) != getattr(policy_duplex, name):
                raise ValueError('KL reference and policy action protocols differ')
        if set(self.duplex.decoder.forbidden_token_ids) != set(policy_duplex.decoder.forbidden_token_ids):
            raise ValueError('KL reference and policy forbidden tokens differ')

    @torch.no_grad()
    def score(self, record, clip, trajectory, config):
        if getattr(self, 'device', torch.device('cpu')).type == 'cuda':
            self.model.to(self.device)
            # MiniCPMODuplex caches its own device separately from model.device.
            # Moving the module alone leaves preprocessed media on the CPU.
            previous_device = self.duplex.device
            self.duplex.device = str(self.device)
            try:
                return self._score(record, clip, trajectory, config).cpu()
            finally:
                self.duplex.device = previous_device
                self.model.to('cpu')
                torch.cuda.empty_cache()
        return self._score(record, clip, trajectory, config)

    def _score(self, record, clip, trajectory, config):
        from train.src.models.streaming import replay
        from train.src.models.cached_replay import cached_hidden
        from train.src.domain.grpo import action_log_probs
        encoded = replay(self.model, self.duplex, record, clip, config, trajectory)
        hidden = cached_hidden(self.model.llm, encoded['segments'])
        positions, ids, turns = encoded['positions'], encoded['token_ids'], encoded['turns']
        if not positions or len(positions) != sum(len(c['actions']) for c in trajectory['chunks']):
            raise ValueError('KL reference action alignment failed')
        scores = []
        for pos, token, turn in zip(positions, ids, turns):
            logits = self.model.llm.lm_head(hidden[pos][None, None, :])[:, -1, :]
            law = action_log_probs(logits, self.duplex.decoder.forbidden_token_ids,
                self.duplex.chunk_eos_token_id, self.duplex.listen_token_id,
                self.duplex.tts_bos_token_id, turn)
            scores.append(law[0, token])
        return torch.stack(scores).detach()
