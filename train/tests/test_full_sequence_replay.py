"""Full causal replay must preserve native probabilities and history gradients."""
import copy
import json
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from train.src.models.cached_replay import cached_hidden
from train.src.models.full_sequence_replay import full_sequence_hidden, checkpoint_decoder_layers


def tiny():
    torch.manual_seed(93)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, attention_dropout=.2,
        _attn_implementation='eager')).double().eval()


def test_full_replay_matches_native_hidden_and_complete_history_gradients():
    first = tiny()
    other = copy.deepcopy(first)
    feeds = [torch.randn(n, 16, dtype=torch.float64, requires_grad=True) for n in (3, 1, 2, 1)]
    clones = [v.detach().clone().requires_grad_() for v in feeds]
    native = cached_hidden(first, feeds)
    actual = full_sequence_hidden(other, torch.cat(clones))
    torch.testing.assert_close(actual, native, atol=1e-10, rtol=1e-10)
    native[-1].square().sum().backward()
    actual[-1].square().sum().backward()
    for a, b in zip(feeds, clones):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9)
        assert b.grad.abs().sum() > 0
    for a, b in zip(first.parameters(), other.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-8)


def test_future_inputs_do_not_change_earlier_outputs_or_create_cache():
    model = tiny()
    x = torch.randn(8, 16, dtype=torch.float64)
    y = x.clone(); y[5:] += 10
    original = model.model.forward
    calls = []
    def check(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    with patch.object(model.model, 'forward', side_effect=check), torch.no_grad():
        a = full_sequence_hidden(model, x)
        b = full_sequence_hidden(model, y)
    torch.testing.assert_close(a[:5], b[:5], atol=0, rtol=0)
    assert all(c['use_cache'] is False and c['past_key_values'] is None for c in calls)


def test_eval_layers_recompute_and_restore_on_success_and_failure():
    model = tiny()
    layer = model.model.layers[0]
    original = layer.forward
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    layer.forward = counted
    x = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    full_sequence_hidden(model, x).sum().backward()
    assert len(calls) == 2
    assert layer.forward is counted and not model.training and not layer.training
    with pytest.raises(RuntimeError, match='restore'):
        with checkpoint_decoder_layers(model.model):
            raise RuntimeError('restore')
    assert layer.forward is counted
    model.train()
    with pytest.raises(ValueError, match='eval'):
        full_sequence_hidden(model, x)


@pytest.mark.parametrize('advantage_mode,aggregation',[
    ('local','unit_weighted'), ('local','token_mean'),
    ('group_relative','unit_weighted'), ('group_relative','token_mean')])
def test_real_graph_multiturn_probabilities_loss_and_media_gradients(advantage_mode, aggregation):
    from train.tests.test_grpo import ReplayGradientTests
    case = ReplayGradientTests()
    try:
        with patch('train.src.models.streaming.user_request', return_value='question'), \
             patch('train.src.models.streaming.system_prompt', return_value='system'):
            graph, record, clip = case.setup_graph(); graph.eval()
            graph.config.update(grpo_advantage_mode=advantage_mode, grpo_loss_aggregation=aggregation)
            clip.chunks = [{}, {}, {}]; clip.question_time = 1
            chunks = [dict(ids=[302], actions=[], idle_wait=True)]
            for reply, ids in enumerate(([301,97,300,98,303,304,305], [301,98,303,304,305])):
                chunks.append(dict(ids=ids, idle_wait=False, actions=[dict(offset=i,
                    turn_open=i>0, reply_id=reply, old_log_prob=0.) for i in range(len(ids)-1)]))
            t = dict(chunks=chunks)
            with torch.no_grad():
                native, _ = graph(record, clip, t)
            actions = [a for c in chunks for a in c['actions']]
            for a, value in zip(actions, native.tolist()):
                a['old_log_prob'] = value
            old = [a['old_log_prob'] for a in actions]
            t.update(advantage=.7, local_credit=dict(version=1, start_second=2, units=[
                dict(kind='reply', reply_id=0, advantage=1., weight=.5, action_count=6,
                     response_index=0, base_advantage=1., blocked_fn_advantage=0.),
                dict(kind='reply', reply_id=1, advantage=-.5, weight=.5, action_count=4,
                     response_index=1, base_advantage=-.5, blocked_fn_advantage=0.)]))
            expected, _ = graph(record, clip, t); expected.backward()
            gradients = {n:p.grad.clone() for n,p in graph.named_parameters() if p.grad is not None}
            graph.zero_grad(set_to_none=True)
            graph.config['grpo_replay_mode'] = 'full_sequence'
            actual, info = graph(record, clip, t); actual.backward()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            assert info['replay_max_logprob_error'] < 1e-5
            for n,p in graph.named_parameters():
                if n in gradients:
                    torch.testing.assert_close(p.grad, gradients[n], atol=1e-5, rtol=1e-4)
            assert graph.model.vpm.weight.grad.abs().sum() > 0
            assert graph.model.apm.weight.grad.abs().sum() > 0
            assert old == [a['old_log_prob'] for a in actions]
    finally:
        case.doCleanups()


def test_mode_configuration_is_opt_in_and_rejects_native_checkpoint_options(tmp_path):
    from train.src.infrastructure.config import load_config, ROOT
    c = json.loads((ROOT/'train/configs/minicpmo_full_grpo_confusion_judge_dp4.json').read_text())
    path = tmp_path/'config.json'
    path.write_text(json.dumps(c))
    assert load_config(path).get('grpo_replay_mode', 'native_kv') == 'native_kv'
    c['grpo_replay_mode'] = 'full_sequence'
    path.write_text(json.dumps(c)); assert load_config(path)['grpo_replay_mode'] == 'full_sequence'
    for key, value in [('grpo_replay_checkpoint_segments',16), ('grpo_replay_checkpoint_feeds',True)]:
        path.write_text(json.dumps(dict(c, **{key:value})))
        with pytest.raises(ValueError, match='without native KV'):
            load_config(path)
