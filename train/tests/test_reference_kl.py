import json
from pathlib import Path

import pytest
import torch

from train.src.infrastructure.config import load_config
from train.src.models.reference_kl import sampled_kl


def trajectory():
    return {'local_credit': {'start_second': 2}, 'chunks': [
        {'actions': [{'reply_id': None}], 'idle_wait': True},
        {'actions': [{'reply_id': None}], 'idle_wait': True},
        {'actions': [{'reply_id': 0}, {'reply_id': 0}], 'idle_wait': False},
        {'actions': [{'reply_id': None}], 'idle_wait': True},
    ]}


def test_reference_kl_zero_and_frozen_reference_gradient():
    current = torch.tensor([-.2, -.5, -1., -1.5, -.8], requires_grad=True)
    reference = current.detach().clone().requires_grad_()
    loss = sampled_kl(current, reference, trajectory())
    assert loss.item() == pytest.approx(0.)
    loss.backward()
    assert reference.grad is None
    assert torch.equal(current.grad, torch.zeros_like(current))


def test_reference_kl_balances_reply_and_wait_units_and_masks_prefix():
    current = torch.zeros(5, requires_grad=True)
    ref = torch.tensor([1., .2, .3, .4, .5])
    f = lambda x: torch.expm1(x) - x
    expected = .5 * ((f(ref[2]) + f(ref[3])) / 2) + .5 * ((f(ref[1]) + f(ref[4])) / 2)
    loss = sampled_kl(current, ref, trajectory())
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert current.grad[0] == 0
    assert all(current.grad[i] < 0 for i in (1, 2, 3, 4))


def test_reference_kl_rejects_misalignment_and_extreme_ratios():
    current = torch.zeros(5)
    with pytest.raises(ValueError, match='align'):
        sampled_kl(current, torch.zeros(4), trajectory())
    with pytest.raises(ValueError, match='safety'):
        sampled_kl(current, torch.full((5,), 21.), trajectory())


def test_kl_configuration_is_opt_in_and_requires_reference(tmp_path):
    root = Path(__file__).resolve().parents[2]
    template = json.loads((root / 'train/configs/minicpmo_full_grpo_tp_full_credit_dp4.json').read_text())
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(template, kl_beta=.01)))
    with pytest.raises(ValueError, match='kl_reference_model_path'):
        load_config(path)
    path.write_text(json.dumps(dict(template, kl_beta=.01, kl_reference_model_path='/tmp/reference')))
    config = load_config(path)
    assert config['kl_beta'] == .01
    assert config['kl_reference_model_path'] == '/tmp/reference'
    path.write_text(json.dumps(dict(template, kl_beta=-1, kl_reference_model_path='/tmp/reference')))
    with pytest.raises(ValueError, match='nonnegative'):
        load_config(path)


def test_update_batch_propagates_reference_and_logs_kl():
    from unittest.mock import patch
    from train.src.training.grpo_judged import update_batch
    from train.tests.test_local_grpo import LocalEngine, make_groups, CONFIG

    class Reference:
        calls = 0
        def score(self, record, clip, trajectory, config):
            self.calls += 1
            size = sum(len(c['actions']) for c in trajectory['chunks'])
            return torch.full((size,), -.3)

    class EngineWithKL(LocalEngine):
        def __call__(self, record, clip, item):
            policy, info = super().__call__(record, clip, item)
            count = sum(len(c['actions']) for c in item['chunks'])
            current = torch.full((count,), -.5) + self.parameter * record['x']
            kl = sampled_kl(current, item['reference_log_probs'], item)
            return policy + .1 * kl, dict(info, kl_surrogate=kl.item(), kl_contribution=.1*kl.item())

    reference = Reference()
    cfg = dict(CONFIG, gradient_accumulation_steps=1, replay_logprob_tolerance=.15,
               grpo_activation_offload=False, kl_beta=.1)
    with patch('train.src.training.grpo_judged.all_true', side_effect=lambda ok, device: ok):
        result = update_batch(EngineWithKL(), make_groups(1), cfg, torch.device('cpu'), reference)
    assert reference.calls == 4
    assert result['kl_surrogate'] > 0
    assert result['kl_contribution'] == pytest.approx(.1 * result['kl_surrogate'])


def test_reference_score_uses_recorded_actions_and_frozen_probabilities(monkeypatch):
    from types import SimpleNamespace
    from train.src.models.reference_kl import FrozenReference
    import train.src.models.streaming as streaming
    import train.src.models.cached_replay as cache
    from train.src.domain.grpo import action_log_probs

    head = torch.nn.Linear(3, 6, bias=False)
    model = SimpleNamespace(llm=SimpleNamespace(lm_head=head))
    duplex = SimpleNamespace(decoder=SimpleNamespace(forbidden_token_ids=[]),
                             chunk_eos_token_id=0, listen_token_id=1, tts_bos_token_id=2)
    policy = object.__new__(FrozenReference)
    policy.model, policy.duplex = model, duplex
    hidden = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    monkeypatch.setattr(streaming, 'replay', lambda *args: dict(
        segments=[hidden], positions=[0, 1], token_ids=[3, 4], turns=[False, True]))
    monkeypatch.setattr(cache, 'cached_hidden', lambda *args: hidden)
    trajectory = {'chunks': [{'actions': [{}, {}]}]}
    scores = policy.score({}, None, trajectory, {})
    expected = torch.stack([action_log_probs(head(hidden[i:i+1]), [], 0, 1, 2, turn)[0, token]
                            for i, (token, turn) in enumerate([(3, False), (4, True)])])
    torch.testing.assert_close(scores, expected)
    assert not scores.requires_grad


def test_cuda_reference_moves_duplex_inputs_and_restores_device(monkeypatch):
    from types import SimpleNamespace
    from train.src.models.reference_kl import FrozenReference

    moves = []
    reference = object.__new__(FrozenReference)
    reference.device = torch.device('cuda:0')
    reference.model = SimpleNamespace(to=lambda device: moves.append(str(device)))
    reference.duplex = SimpleNamespace(device='cpu')
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    def score(*args):
        assert reference.duplex.device == 'cuda:0'
        return torch.tensor([1.])
    reference._score = score
    torch.testing.assert_close(reference.score({}, None, {}, {}), torch.tensor([1.]))
    assert moves == ['cuda:0', 'cpu']
    assert reference.duplex.device == 'cpu'
