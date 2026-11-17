import pytest
import torch


def test_action_gradient_monitor_keeps_objective_and_reports_components(monkeypatch):
    from train.src.infrastructure.grpo_gradient_monitor import install
    from train.src.infrastructure.tracking import Tracking
    from train.src.models import local_policy, reference_kl
    from train.src.training import grpo_judged

    monkeypatch.setattr(local_policy, 'local_policy_loss', lambda current: (current.sum(),))
    monkeypatch.setattr(reference_kl, 'sampled_kl', lambda current: current.square().sum())
    def update(engine, groups, config, device, reference=None):
        current = torch.tensor([1., 2.], requires_grad=True)
        policy = local_policy.local_policy_loss(current)[0]
        kl = reference_kl.sampled_kl(current)
        loss = policy + config['kl_beta'] * kl
        loss.backward()
        assert current.grad.tolist() == pytest.approx([1.2, 1.4])
        return {'loss': loss.item()}
    monkeypatch.setattr(grpo_judged, 'update_batch', update)
    monkeypatch.setattr(Tracking, 'log_train', lambda self, record: None)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    install()
    result = grpo_judged.update_batch(None, [], {'kl_beta': .1}, torch.device('cpu'))
    assert result['loss'] == pytest.approx(3.5)
    assert result['action_grad_policy_rms_norm'] == pytest.approx(2**.5)
    assert result['action_grad_kl_rms_norm'] == pytest.approx(.2 * 5**.5)
    assert result['action_grad_cosine'] == pytest.approx(3 / 10**.5)
    assert result['action_grad_trajectories'] == 1
