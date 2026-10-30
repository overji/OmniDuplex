"""Per-QA hierarchical weights; no logits or trainable parameters are stored here."""
import torch


def event_weights(labels, groups, roles, events, config, listen_id, speak_id):
    """Inputs have already received the causal shift; return one CE weight/token.

    Gate leaves use token means; structure and content leaves use equal event
    means, then equal token means within each event. Missing branches renormalize.
    """
    if roles is None or events is None or roles.shape != labels.shape or events.shape != labels.shape:
        raise ValueError('event_balanced requires aligned event_roles and event_ids')
    valid = labels != -100
    decision = valid & (groups == 1)
    if bool((valid & ((groups < 1) | (groups > 3))).any()):
        raise ValueError('Invalid supervised loss group')
    if bool((decision & ((roles < 1) | (roles > 7))).any()) or bool((valid & (groups != 1) & (roles != 0)).any()):
        raise ValueError('Invalid event role for supervised token')
    idle = decision & (roles <= 2)
    onset = decision & ((roles == 3) | (roles == 4))
    if bool((idle & ((labels != listen_id) | (events != -1))).any()) or bool((onset & (labels != speak_id)).any()):
        raise ValueError('Event roles disagree with silence/onset labels')
    if bool((valid & ~idle & (events < 0)).any()):
        raise ValueError('Response tokens require an event ID')
    if bool((decision & (roles == 5) & (labels != speak_id)).any()):
        raise ValueError('Continuation requires speak token')
    if bool((onset & (((roles == 3) & (events != 0)) | ((roles == 4) & (events <= 0)))).any()):
        raise ValueError('First/later onset disagrees with event ID')

    def leaf(mask, per_event=False):
        result = torch.zeros_like(labels, dtype=torch.float32)
        count = int(mask.sum())
        if not count:
            return None
        if per_event:
            ids, counts = torch.unique(events[mask], return_counts=True)
            for event, n in zip(ids, counts):
                result[mask & (events == event)] = 1. / (len(ids) * n)
        else:
            result[mask] = 1. / count
        return result

    def combine(branches):
        active = [(weight, value) for weight, value in branches if weight > 0 and value is not None]
        if not active:
            return None
        total = sum(weight for weight, _ in active)
        return sum(weight / total * value for weight, value in active)

    leaves = {r: leaf(decision & (roles == r), per_event=r >= 5) for r in range(3, 8)}
    # Before/after roles are diagnostic only: every idle token has equal weight.
    silence = leaf(idle)
    onset = combine([(1, leaves[3]), (1, leaves[4])])
    gate = combine([(config.get('silent_weight', .5), silence),
                    (config.get('onset_weight', .5), onset)])
    structure = combine([(1, leaves[r]) for r in (5, 6, 7)])
    decision_weights = combine([(config.get('gate_weight', .8), gate),
                                (config.get('structure_weight', .2), structure)])
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for scale, value in ((config['decision_weight'], decision_weights),
                         (config['content_weight'], leaf(valid & (groups == 2), True)),
                         (config['call_weight'], leaf(valid & (groups == 3), True))):
        if value is not None:
            weights += scale * value
    return weights
