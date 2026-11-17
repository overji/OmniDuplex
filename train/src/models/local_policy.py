"""Independent advantage and aggregation choices for judged GRPO ablations."""
import math

import torch

from train.src.domain.grpo import policy_loss


def action_groups(trajectory):
    """Index sampled actions once, preserving replay order and question masking."""
    replies, idle, controls, eligible = {}, {}, {}, []
    offset = 0
    start = trajectory['local_credit'].get('start_second', 1)
    for second, chunk in enumerate(trajectory['chunks'], 1):
        for index, action in enumerate(chunk['actions'], offset):
            if second < start:
                continue
            eligible.append(index)
            owner = action['reply_id']
            if owner is not None:
                replies.setdefault(owner, []).append(index)
            elif chunk['idle_wait']:
                idle.setdefault(second, []).append(index)
            else:
                controls.setdefault(second, []).append(index)
        offset += len(chunk['actions'])
    return replies, idle, controls, eligible, offset


def unit_groups(unit, replies, idle, controls):
    """Return token spans and their within-unit means, independent of reward."""
    if unit['kind'] == 'reply':
        positions = replies.get(unit['reply_id'], [])
        if len(positions) != unit['action_count']:
            raise ValueError('Reply credit does not cover its sampled actions')
        groups = [positions]
    elif unit['kind'] in ('fn', 'wait', 'control'):
        seconds = unit['seconds']
        if not seconds or len(set(seconds)) != len(seconds):
            raise ValueError('Local waiting unit requires distinct seconds')
        lookup = controls if unit['kind'] == 'control' else idle
        groups = [lookup.get(second, []) for second in seconds]
    else:
        raise ValueError('Unknown local policy unit')
    if any(not positions for positions in groups):
        raise ValueError('Local policy unit has no sampled actions')
    return groups


def local_policy_loss(current, old, trajectory, clip, *, advantage_mode='local', aggregation='unit_weighted'):
    if current.numel() == 0 or current.shape != old.shape:
        raise ValueError('Local policy requires aligned nonempty sampled probabilities')
    credit = trajectory['local_credit']
    if credit.get('version') != 1:
        raise ValueError('Unsupported local credit version')
    if advantage_mode not in ('local', 'group_relative') or aggregation not in ('token_mean', 'unit_weighted'):
        raise ValueError('Invalid GRPO advantage or aggregation mode')
    replies, idle, controls, eligible, count = action_groups(trajectory)
    if count != current.numel() or not eligible:
        raise ValueError('Local units and replay action counts disagree')
    shared = trajectory.get('advantage') if advantage_mode == 'group_relative' else None
    if advantage_mode == 'group_relative' and (shared is None or not math.isfinite(shared)):
        raise ValueError('Group-relative objective requires a finite trajectory advantage')
    ratio = (current-old).exp()
    if aggregation == 'token_mean':
        if advantage_mode == 'group_relative':
            advantage = shared
        else:
            values = [0.] * count
            for unit in credit['units']:
                value = unit['advantage']
                if not math.isfinite(value):
                    raise ValueError('Invalid local advantage')
                for positions in unit_groups(unit, replies, idle, controls):
                    for index in positions:
                        values[index] += value
            advantage = current.new_tensor(values)[eligible]
        loss, _ = policy_loss(current[eligible], old[eligible], advantage, clip)
        return loss, ratio.detach(), {'local_loss_token_mean': loss.detach().item()}

    # Positive/negative coefficients stay separate because PPO clipping depends
    # on the sign. Unit weights contain all event/time normalization exactly once.
    names = ('reply', 'fp', 'fn', 'wait', 'control')
    coefficients = {name: [[0.]*count, [0.]*count] for name in names}
    covered = set()
    for unit in credit['units']:
        advantage = shared if advantage_mode == 'group_relative' else unit['advantage']
        weight = unit['weight']
        if not math.isfinite(advantage) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid local policy unit')
        groups = unit_groups(unit, replies, idle, controls)
        name = unit['kind']
        if name == 'reply' and unit['response_index'] is None:
            name = 'fp'
        values = coefficients[name][int(advantage < 0)]
        for positions in groups:
            covered.update(positions)
            value = advantage*weight/(len(groups)*len(positions))
            for index in positions:
                values[index] += value
    if advantage_mode == 'group_relative' and covered != set(eligible):
        raise ValueError('Unit-weighted group advantage requires complete structural action coverage')
    bounded = ratio.clamp(1-clip, 1+clip)
    positive, negative = torch.minimum(ratio, bounded), torch.maximum(ratio, bounded)
    losses = {}
    for name in names:
        values = current.new_tensor(coefficients[name])
        losses[name] = -(positive*values[0] + negative*values[1]).sum()
    loss = sum(losses.values())
    blocked_delta = current.detach().new_zeros(())
    if advantage_mode == 'local':
        # Diagnostic delta only; blocked FN already enters the reply advantage.
        with torch.no_grad():
            for unit in credit['units']:
                if unit['kind'] != 'reply' or not unit['blocked_fn_advantage']:
                    continue
                indices = replies[unit['reply_id']]
                combined, _ = policy_loss(current[indices], old[indices], unit['advantage'], clip)
                base, _ = policy_loss(current[indices], old[indices], unit['base_advantage'], clip)
                blocked_delta += (combined-base)*unit['weight']
    info = {f'local_loss_{name}': value.detach().item() for name,value in losses.items()}
    info['local_loss_blocked_fn'] = blocked_delta.item()
    return loss, ratio.detach(), info
