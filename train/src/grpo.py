"""Group-relative policy math and MiniCPM's two-stage native action law."""
import torch
import torch.nn.functional as F


def action_log_probs(logits, forbidden, eos, listen, speak, turn_open):
    """Marginalize native EOS coin flip and listen -> tts_bos remapping.

    temperature=1, no repetition/top-k/top-p/listen scaling. Returns the
    distribution of the token actually fed back into the model, not the latent
    draw that native generation may remap while an utterance is active.
    """
    z = logits.float()
    raw = F.log_softmax(z, -1)
    mask = torch.zeros(z.shape[-1], dtype=torch.bool, device=z.device)
    mask[list(set(forbidden) | {eos})] = True
    conditional = F.log_softmax(z.masked_fill(mask, -torch.inf), -1)
    # log(1-p_eos), stably, without subtracting a nearly-one probability.
    non_eos = z.clone()
    non_eos[..., eos] = -torch.inf
    log_continue = torch.logsumexp(non_eos, -1) - torch.logsumexp(z, -1)
    combined = conditional + log_continue.unsqueeze(-1)
    eos_mask = torch.arange(z.shape[-1], device=z.device) == eos
    combined = torch.where(eos_mask, raw[..., eos, None], combined)
    if isinstance(turn_open, bool):
        turn_open = torch.full(z.shape[:-1], turn_open, device=z.device, dtype=torch.bool)
    if turn_open.any():
        # Native replaces a sampled listen with tts_bos inside an unfinished turn.
        merged = torch.logaddexp(combined[..., speak], combined[..., listen])
        columns = torch.arange(z.shape[-1], device=z.device)
        combined = torch.where(turn_open.unsqueeze(-1) & (columns == speak), merged.unsqueeze(-1), combined)
        combined = torch.where(turn_open.unsqueeze(-1) & (columns == listen), -torch.inf, combined)
    return combined


def advantages(rewards):
    scores = torch.as_tensor(rewards, dtype=torch.float32)
    std = scores.std(unbiased=False)
    if std <= 1e-8:
        return torch.zeros_like(scores), True
    return (scores - scores.mean()) / (std + 1e-4), False


def policy_loss(current, old, advantage, clip=.2):
    if current.numel() == 0 or current.shape != old.shape:
        raise ValueError('GRPO requires aligned, nonempty action log probabilities')
    ratio = (current - old).exp()
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -objective.mean(), ratio.detach()
