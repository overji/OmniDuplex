"""Opt-in action-log-probability gradient diagnostics for local GRPO.

The measured gradients are with respect to sampled action log probabilities,
not model parameters. Monitoring does not alter the loss or optimizer.
"""
import math

import torch


def install():
    from train.src.models import local_policy, reference_kl
    from train.src.training import grpo_judged
    from train.src.infrastructure.tracking import Tracking

    state = {'active': False, 'policy': None, 'stats': None}
    original_policy = local_policy.local_policy_loss
    original_kl = reference_kl.sampled_kl
    original_update = grpo_judged.update_batch
    original_tracking = Tracking.log_train

    def policy(current, *args, **kwargs):
        result = original_policy(current, *args, **kwargs)
        if state['active']:
            gradient, = torch.autograd.grad(result[0], current, retain_graph=True)
            state['policy'] = gradient.detach().float()
        return result

    def kl(current, *args, **kwargs):
        result = original_kl(current, *args, **kwargs)
        if state['active']:
            gradient, = torch.autograd.grad(result, current, retain_graph=True)
            p = state.pop('policy')
            if p is None:
                raise RuntimeError('Missing paired policy gradient')
            k = gradient.detach().float() * state['beta']
            stats = state['stats']
            stats[0] += p.square().sum().item()
            stats[1] += k.square().sum().item()
            stats[2] += (p * k).sum().item()
            stats[3] += 1
            state['policy'] = None
        return result

    def update(engine, groups, config, device, reference=None):
        state.update(active=True, beta=config['kl_beta'], policy=None, stats=[0., 0., 0., 0.])
        try:
            result = original_update(engine, groups, config, device, reference)
            totals = torch.tensor(state['stats'], device=device, dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
            policy_sq, kl_sq, dot, count = totals.tolist()
            if count:
                pnorm, knorm = math.sqrt(policy_sq/count), math.sqrt(kl_sq/count)
                result.update(action_grad_policy_rms_norm=pnorm, action_grad_kl_rms_norm=knorm,
                              action_grad_kl_policy_ratio=knorm/pnorm if pnorm else 0.,
                              action_grad_cosine=dot/math.sqrt(policy_sq*kl_sq) if policy_sq and kl_sq else 0.,
                              action_grad_trajectories=int(count))
            return result
        finally:
            state.update(active=False, policy=None, stats=None)

    def tracking(self, record):
        original_tracking(self, record)
        values = {'train/step': record['step']}
        values.update({'train/' + key: value for key, value in record.items()
                       if key.startswith('action_grad_') and isinstance(value, (int, float))})
        if len(values) > 1:
            if self.tensorboard is not None:
                self.tensorboard.emit(values)
            if self.run is not None:
                self.run.log(values)

    local_policy.local_policy_loss = policy
    reference_kl.sampled_kl = kl
    grpo_judged.update_batch = update
    Tracking.log_train = tracking
