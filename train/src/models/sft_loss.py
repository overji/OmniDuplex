"""SFT CE with explicit fresh-onset balancing and additive diagnostics."""
import torch
from torch.utils.checkpoint import checkpoint
from train.src.domain.loss_metrics import METRIC_SIZE


def grouped_loss(head, hidden, labels, groups, config, listen_id, speak_id, decision_roles=None, event_roles=None, event_ids=None):
    """Average within each QA; vocabulary projection remains chunked.

    Roles are aligned with target labels, before the causal one-token shift.
    Metrics are detached and never mutated by checkpoint recomputation.
    """
    loss = hidden.sum() * 0
    metrics = torch.zeros(METRIC_SIZE, device=hidden.device, dtype=torch.float64)
    balanced = config.get('decision_loss_mode', 'token_mean') == 'balanced_gate'
    if balanced and decision_roles is None:
        raise ValueError('Balanced gate loss requires explicit decision_roles')
    event_mode = config.get('decision_loss_mode') == 'event_balanced'
    event_ce_weights = None
    if event_mode:
        from train.src.models.event_loss import event_weights
        event_ce_weights = event_weights(labels[1:], groups[1:],
            event_roles[1:] if event_roles is not None else None,
            event_ids[1:] if event_ids is not None else None, config, listen_id, speak_id)
    role_weights = {}
    if balanced:
        valid = (labels[1:] != -100) & (groups[1:] == 1)
        roles = decision_roles[1:]
        if bool((valid & ((roles < 1) | (roles > 3))).any()):
            raise ValueError('Every decision target requires a valid role')
        if bool((valid & (((roles == 1) & (labels[1:] != listen_id)) |
                          ((roles == 2) & (labels[1:] != speak_id)))).any()):
            raise ValueError('Decision roles disagree with listen/onset labels')
        counts = {r: int((valid & (roles == r)).sum()) for r in (1, 2, 3)}
        gate = {r: config.get(k, .5) for r, k in ((1, 'silent_weight'), (2, 'onset_weight'))
                if counts[r] and config.get(k, .5) > 0}
        gw = config.get('gate_weight', .8) if gate else 0.
        sw = config.get('structure_weight', .2) if counts[3] else 0.
        outer = gw + sw
        if outer:
            for r, weight in gate.items():
                role_weights[r] = gw / outer * weight / sum(gate.values()) / counts[r]
            if sw:
                role_weights[3] = sw / outer / counts[3]
    active = False
    for group, key in ((1, 'decision_weight'), (2, 'content_weight'), (3, 'call_weight')):
        positions = torch.where((labels[1:] != -100) & (groups[1:] == group))[0]
        if not len(positions) or not config[key]:
            continue
        if group == 1 and balanced and not role_weights:
            continue
        if event_mode and not bool((event_ce_weights[positions] > 0).any()):
            continue
        active = True
        for ix in positions.split(config['loss_projection_chunk']):
            def project(h, targets, decision):
                logits = head(h).float()
                ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
                total = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
                with torch.no_grad():
                    tokens = ce.detach() if decision else logits.new_empty(0)
                return total, tokens, ce
            part, token_ce, ce = checkpoint(project, hidden[ix], labels[ix + 1], group == 1,
                                        use_reentrant=False)
            # Historical mode preserves its reduction and weighting order.
            if event_mode:
                weights = event_ce_weights[ix]
                contribution = (ce * weights).sum()
            elif group == 1 and balanced:
                weights = torch.zeros_like(ce)
                for role, weight in role_weights.items():
                    weights = weights + (decision_roles[ix + 1] == role) * weight
                contribution = config[key] * (ce * weights).sum()
            else:
                contribution = config[key] * part / len(positions)
            loss = loss + contribution
            with torch.no_grad():
                component = {1: 5, 2: 3, 3: 4}[group]
                metrics[component] += part.detach().double()
                metrics[6 + component] += len(ix)
                metrics[11 + group] += contribution.detach().double()
                if event_mode and group == 1:
                    fine = event_roles[ix + 1]
                    weighted = ce.detach().double() * weights.detach().double()
                    for role in range(1, 8):
                        mask = fine == role
                        metrics[22 + role] += ce.detach()[mask].double().sum()
                        metrics[29 + role] += mask.sum()
                        metrics[36 + role] += weighted[mask].sum()
                    metrics[21] += weighted[fine <= 4].sum()
                    metrics[22] += weighted[fine >= 5].sum()
                if group == 1:
                    target = labels[ix + 1]
                    masks = (target == listen_id, target == speak_id,
                             (target != listen_id) & (target != speak_id))
                    for index, mask in enumerate(masks):
                        metrics[index] += token_ce[mask].double().sum()
                        metrics[6 + index] += mask.sum()
                    if decision_roles is not None:
                        role = decision_roles[ix + 1]
                        diagnostic_masks = (role == 2, (role == 3) & (target == speak_id), role == 3)
                        for index, mask in enumerate(diagnostic_masks):
                            metrics[15 + index] += token_ce[mask].double().sum()
                            metrics[18 + index] += mask.sum()
                        if balanced:
                            weighted = config[key] * token_ce.double() * weights.detach().double()
                            metrics[21] += weighted[role != 3].sum()
                            metrics[22] += weighted[role == 3].sum()
    if not active:
        raise ValueError('No supervised tokens with nonzero loss weight')
    return loss, metrics.cpu().tolist()
