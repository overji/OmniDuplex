import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from train.src.domain.grpo import action_log_probs, advantages, policy_loss
from train.src.domain.rewards import reward, text_f1, value_score
from train.src.models.graph import TrainGraph
from train.src.infrastructure.media import Clip
from train.tests.helpers import ByteTokenizer


class GRPOMathTests(unittest.TestCase):
    def test_grpo_config_enforces_native_replay_assumptions(self):
        from train.src.infrastructure.config import ROOT, load_config
        config = load_config(ROOT / 'train/configs/minicpmo_full_grpo.json')
        self.assertEqual(config['agent_output_protocol'], 'task_start_token_sft_v1')
        self.assertEqual(config['group_size'], 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for key, value in (('group_size', 1), ('temperature', .7), ('kl_beta', .1),
                               ('clip_epsilon', 0), ('replay_logprob_tolerance', 0)):
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(config, **{key: value})))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_saved_cpu_parameter_views_deduplicate_without_changing_gradients(self):
        from train.src.infrastructure.offload import saved_on_cpu
        layer = torch.nn.Linear(8, 4)
        x = torch.randn(3, 8, requires_grad=True)
        expected = sum(layer(x).square().mean() for _ in range(4))
        expected.backward()
        gradients = [p.grad.clone() for p in layer.parameters()]
        x_grad = x.grad.clone()
        layer.zero_grad(); x.grad = None
        with saved_on_cpu() as stats:
            loss = sum(layer(x).square().mean() for _ in range(4))
            loss.backward()
        self.assertGreaterEqual(stats['parameter_cache_hits'], 3)
        for parameter, expected in zip(layer.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected)
        torch.testing.assert_close(x.grad, x_grad)
        with self.assertRaisesRegex(RuntimeError, 'CPU budget'):
            with saved_on_cpu(budget_gib=1e-12):
                layer(x).sum().backward()

    def test_activation_offload_preserves_padded_and_broadcast_strides(self):
        from train.src.infrastructure.offload import saved_on_cpu, copy_strided
        padded = torch.randn(3, 8)[:, :6]
        restored = copy_strided(padded, 'cpu')
        self.assertEqual(restored.stride(), padded.stride())
        torch.testing.assert_close(restored, padded)
        expanded = torch.randn(1, 3).expand(4, 3)
        restored = copy_strided(expanded, 'cpu')
        self.assertEqual(restored.stride(), expanded.stride())
        torch.testing.assert_close(restored, expanded)
        class Check(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(torch.randn(3, 8)[:, :6])
                return x.sum()
            @staticmethod
            def backward(ctx, grad):
                assert ctx.saved_tensors[0].stride() == (8, 1)
                return grad.expand(3)
        x = torch.ones(3, requires_grad=True)
        with saved_on_cpu():
            Check.apply(x).backward()
        torch.testing.assert_close(x.grad, torch.ones(3))
    def test_native_marginal_distribution_and_remapped_listen(self):
        z = torch.tensor([[.1, .3, .7, -.2, .4]], requires_grad=True)
        eos, listen, speak = 0, 1, 2
        raw = z.softmax(-1)[0]
        conditional = z.masked_fill(torch.tensor([[True, False, False, False, True]]), -torch.inf).softmax(-1)[0]
        expected = (1 - raw[eos]) * conditional
        expected = expected.clone(); expected[eos] = raw[eos]
        actual = action_log_probs(z, [0, 4], eos, listen, speak, False).exp()[0]
        torch.testing.assert_close(actual, expected)
        expected = expected.clone(); expected[speak] += expected[listen]; expected[listen] = 0
        actual = action_log_probs(z, [0, 4], eos, listen, speak, True)
        torch.testing.assert_close(actual.exp()[0], expected)
        (-actual[0, speak]).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum(), 0)

    def test_group_advantage_and_clipped_loss(self):
        values, skip = advantages([1., 1., 1., 1.])
        self.assertTrue(skip); self.assertEqual(values.sum(), 0)
        values, skip = advantages([0., 0., 1., 1.])
        self.assertFalse(skip); self.assertAlmostEqual(values.mean().item(), 0)
        old = torch.tensor([-.5, -.5])
        current = (old + torch.tensor([.5, -.5])).requires_grad_()
        loss, ratio = policy_loss(current, old, 1., .2)
        expected = -torch.minimum(ratio, ratio.clamp(.8, 1.2)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward(); self.assertEqual(current.grad[0], 0)
        self.assertLess(current.grad[1], 0)


class RewardTests(unittest.TestCase):
    def sample(self):
        return dict(mode='rtd', responses=[dict(window_start='00:01', window_end='00:02', response='Red light')])

    def test_text_matching_and_timing(self):
        self.assertEqual(text_f1('RED light!', 'red light'), 1.)
        sample = self.sample()
        good = dict(onset_seconds=1, response_text='Red light', text='Red light')
        self.assertEqual(reward(sample, [good], {})['total'], 1.)
        self.assertEqual(reward(sample, [], {})['total'], 0.)
        late = dict(good, onset_seconds=3)
        self.assertEqual(reward(sample, [late], {})['total'], 0.)
        self.assertLess(reward(sample, [good, good], {})['total'], 1.)

    def test_agent_negative_invalid_and_structured_arguments(self):
        sample = self.sample(); sample.update(mode='agent', sample_type='missing_tool')
        event = dict(onset_seconds=1, response_text='Red light', text='Red light', tool_calls=[])
        self.assertEqual(reward(sample, [event], {})['call'], 1.)
        attempted = dict(event, agent_attempted=True, agent_parse_status='truncated')
        r = reward(sample, [attempted], {})
        self.assertEqual(r['call'], 0.)
        self.assertEqual(r['invalid_calls'], 1)
        self.assertEqual(value_score({'instruction': 'search red', 'n': 1}, {'instruction': 'search red', 'n': 2}), .5)
        sample.pop('sample_type')
        call = dict(type='function', function=dict(name='task_start', arguments={'instruction': 'search'}))
        sample['responses'][0].update(role='assistant', tool_calls=[call])
        event.update(tool_calls=[call], agent_parse_status='structured')
        self.assertEqual(reward(sample, [event], {})['total'], 1.)


class TinyMedia(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=320, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
        self.vpm, self.apm = torch.nn.Linear(3, 16), torch.nn.Linear(3, 16)
        self.resampler, self.audio_projection_layer = torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)
        self.tts = torch.nn.Linear(3, 3)

    def init_streaming_processor(self):
        self.audio_past_key_values = None


class TinyDuplex:
    def __init__(self, model):
        self.model = model
        self.tokenizer = ByteTokenizer()
        self.processor = SimpleNamespace(process_image=lambda *a, **k: None,
                                         process_audio_streaming=lambda *a, **k: None)
        self.decoder = SimpleNamespace(forbidden_token_ids=[304])
        self.listen_token_id, self.chunk_eos_token_id, self.tts_bos_token_id = 302, 304, 301

    def _reset_streaming_state(self):
        pass

    @torch.no_grad()
    def streaming_prefill(self, text_list=None, **kwargs):
        if text_list:
            self.decoder.feed(self.decoder.embed_token(self.tokenizer.encode(text_list[0])))
        else:
            self.decoder.feed(self.model.resampler(self.model.vpm(torch.ones(1, 3))))
            self.decoder.feed(self.model.audio_projection_layer(self.model.apm(torch.ones(1, 3))))
        return dict(success=True)


class ReplayGradientTests(unittest.TestCase):
    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    def test_atomic_task_start_is_replayed_and_trained_as_an_action(self, *_):
        graph, record, clip = self.setup_graph()
        graph.eval()
        # The separator is a sampled action and also conditions the instruction.
        ids = [301, 97, 300, 98, 303, 304, 305]
        trajectory = dict(chunks=[dict(ids=ids, actions=[
            dict(offset=i, turn_open=i > 0, old_log_prob=0.) for i in range(4)])])
        with torch.no_grad():
            current, _ = graph(record, clip, trajectory)
        for action, lp in zip(trajectory['chunks'][0]['actions'], current.tolist()):
            action['old_log_prob'] = lp
        trajectory['advantage'] = 1.
        loss, metrics = graph(record, clip, trajectory)
        self.assertLess(metrics['replay_max_logprob_error'], 1e-5)
        loss.backward()
        self.assertGreater(graph.model.llm.get_input_embeddings().weight.grad[300].abs().sum(), 0)
        self.assertGreater(graph.model.llm.lm_head.weight.grad[300].abs().sum(), 0)

    def test_cached_audio_restores_streaming_processor_state(self):
        class Processor:
            def __init__(self):
                self.state, self.calls = 0, 0
            def process_image(self, *args):
                return None
            def process_audio_streaming(self, *args):
                self.calls += 1; self.state += 1
                return {'feature': self.state}
            def get_streaming_snapshot(self):
                return {'state': self.state}
            def restore_streaming_snapshot(self, snapshot):
                self.state = snapshot['state']
        clip = Clip.__new__(Clip); clip.processed = {}
        processor = Processor()
        for repeat in range(2):
            processor.state = 0
            with clip.preprocessing_cache(processor):
                self.assertEqual(processor.process_audio_streaming()['feature'], 1)
                self.assertEqual(processor.state, 1)
                self.assertEqual(processor.process_audio_streaming()['feature'], 2)
                self.assertEqual(processor.state, 2)
        self.assertEqual(processor.calls, 2)

    def setup_graph(self):
        model = TinyMedia(); model.requires_grad_(False)
        duplex = TinyDuplex(model)
        config = dict(max_sequence_tokens=1024, max_speak_tokens_per_chunk=20,
            agent_output_protocol='task_start_token_sft_v1', supervise_silence=False,
            decision_weight=1., content_weight=1., call_weight=1., loss_projection_chunk=2)
        from pathlib import Path
        import tempfile
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        Path(directory.name, 'video.mp4').touch()
        config.update(videos_dir=directory.name, max_prefix_seconds=32)
        graph = TrainGraph(model, duplex, config)
        ref = dict(response='a', window_start='00:01', window_end='00:02')
        record = dict(sample_index=0, sample=dict(mode='rtd', question_time='00:00', video='video.mp4', responses=[ref]),
                      targets=[dict(response_index=0, onset_seconds=1, reference=ref)])
        from contextlib import nullcontext
        clip = SimpleNamespace(chunks=[{}], question_time=0, preprocessing_cache=lambda p: nullcontext())
        return graph, record, clip

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    def test_full_sft_updates_all_modalities_and_keeps_tts_frozen(self, *_):
        graph, record, clip = self.setup_graph()
        before = {k: v.detach().clone() for k, v in graph.model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in graph.parameters() if p.requires_grad], lr=.01)
        graph(record, clip).backward()
        for module in ('llm', 'vpm', 'apm', 'resampler', 'audio_projection_layer'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(graph.model, module).parameters()), module)
        self.assertTrue(all(p.grad is None for p in graph.model.tts.parameters()))
        optimizer.step()
        for module in ('llm', 'vpm', 'apm', 'resampler', 'audio_projection_layer'):
            self.assertTrue(any(not torch.equal(before[k], p) for k,p in graph.model.named_parameters() if k.startswith(module + '.')), module)

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    def test_controlled_grpo_replay_and_nonzero_full_parameter_gradients(self, *_):
        graph, record, clip = self.setup_graph()
        graph.eval()
        trajectory = dict(chunks=[dict(ids=[301, 97, 303, 304, 305], actions=[
            dict(offset=0, turn_open=False, old_log_prob=0.),
            dict(offset=1, turn_open=True, old_log_prob=0.)])])
        with torch.no_grad():
            current, _ = graph(record, clip, trajectory)
        for action, lp in zip(trajectory['chunks'][0]['actions'], current.tolist()):
            action['old_log_prob'] = lp
        trajectory['advantage'] = 1.
        loss, metrics = graph(record, clip, trajectory)
        self.assertLess(metrics['replay_max_logprob_error'], 1e-5)
        self.assertAlmostEqual(metrics['ratio_mean'], 1., places=5)
        loss.backward()
        for module in ('llm', 'vpm', 'apm', 'resampler', 'audio_projection_layer'):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                for p in getattr(graph.model, module).parameters()), module)


if __name__ == '__main__':
    unittest.main()
