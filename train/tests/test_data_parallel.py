import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from train.src.domain.batching import epoch_slots
from train.src.models.graph import TrainGraph
from train.src.training.data_parallel import step
from train.tests.helpers import TinyMedia, TinyDuplex


class Engine:
    def __init__(self, graph):
        self.module, self.calls = graph, 0
    def __call__(self, *args):
        self.calls += 1
        return self.module(*args)
    def backward(self, loss):
        loss.backward()


class ParallelTests(unittest.TestCase):
    def test_epoch_slots_cover_each_qa_once_with_padding(self):
        slots = [epoch_slots(11, 42, 0, 6, rank, 1, True) for rank in range(6)]
        flat = [x for part in slots for x in part if x is not None]
        self.assertEqual(sorted(flat), list(range(11)))
        self.assertTrue(all(len(x) == 2 for x in slots))
        self.assertEqual(slots, [epoch_slots(11, 42, 0, 6, r, 1, True) for r in range(6)])
        self.assertNotEqual(slots[0], epoch_slots(11, 42, 1, 6, 0, 1, True))

    def test_length_buckets_preserve_coverage_and_group_similar_qa(self):
        lengths = [1, 40, 3, 50, 8, 100, 2, 90, 4, 30, 6, 25]
        slots = [epoch_slots(12, 42, 0, 3, rank, 1, True, lengths=lengths, bucket_size=12) for rank in range(3)]
        interleaved = [slots[rank][step] for step in range(4) for rank in range(3)]
        self.assertEqual(sorted(interleaved), list(range(12)))
        self.assertEqual([lengths[i] for i in interleaved], sorted(lengths, reverse=True))

    def test_accumulation_covers_tail_without_repeating_qa(self):
        slots = [epoch_slots(3996, 42, 0, 4, rank, 4, True) for rank in range(4)]
        self.assertTrue(all(len(part) == 1000 for part in slots))
        self.assertTrue(all(len(part) % 4 == 0 for part in slots))
        self.assertEqual(sorted(i for part in slots for i in part if i is not None), list(range(3996)))
        self.assertTrue(all(part[-1] is None for part in slots))

    def test_data_parallel_config_accepts_accumulation(self):
        from train.src.infrastructure.config import ROOT, load_config
        config = load_config(ROOT / 'train/configs/minicpmo_full_sft_dp.json')
        self.assertEqual(config['gradient_accumulation_steps'], 4)

    def test_tail_config_requires_silence_supervision_and_nonnegative_seconds(self):
        import json
        from train.src.infrastructure.config import ROOT, load_config
        config = load_config(ROOT / 'train/configs/minicpmo_full_sft_dp.json')
        self.assertEqual(config['post_response_silence_seconds'], 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for overrides in ({'post_response_silence_seconds': -1},
                              {'post_response_silence_seconds': True},
                              {'supervise_silence': False}):
                path.write_text(json.dumps({**config, **overrides}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def setup_graph(self, directory):
        Path(directory, 'video.mp4').touch()
        config = dict(videos_dir=directory, max_prefix_seconds=32, max_sequence_tokens=1024,
                      max_speak_tokens_per_chunk=20, agent_output_protocol='task_start_text_sft_v1',
                      supervise_silence=True, decision_weight=1., content_weight=1., call_weight=1.,
                      loss_projection_chunk=2, sft_activation_offload=True)
        model = TinyMedia()
        graph = TrainGraph(model, TinyDuplex(model), config).eval()
        ref = dict(response='a', window_start='00:01', window_end='00:02')
        record = dict(sample_index=0, sample=dict(mode='rtd', question_time='00:00', video='video.mp4', responses=[ref]),
                      targets=[dict(response_index=0, onset_seconds=1, reference=ref)])
        clip = SimpleNamespace(chunks=[{}], horizon=1, duration=2, question_time=0,
                               preprocessing_cache=lambda p: nullcontext())
        return config, graph, record, clip

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    @patch('torch.distributed.all_reduce')
    def test_single_pass_matches_full_sft_loss_and_gradients(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            config, graph, record, clip = self.setup_graph(directory)
            reference = copy.deepcopy(graph)
            expected = reference(record, clip)
            expected.backward()
            engine = Engine(graph)
            with patch('train.src.training.data_parallel.Clip', return_value=clip):
                value, info = step(engine, record, config, torch.device('cpu'), 1)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(info['valid_qa'], 1)
            self.assertAlmostEqual(value, expected.item(), places=5)
            for (name, parameter), (_, other) in zip(graph.named_parameters(), reference.named_parameters()):
                if other.grad is not None:
                    torch.testing.assert_close(parameter.grad, other.grad, msg=name)

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    @patch('torch.distributed.all_reduce')
    def test_short_video_caps_optional_tail_without_excluding_qa(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            config, graph, record, clip = self.setup_graph(directory)
            config['post_response_silence_seconds'] = 10
            clip.horizon, clip.chunks = 2, [{}, {}]
            engine = Engine(graph)
            with patch('train.src.training.data_parallel.Clip', return_value=clip):
                value, info = step(engine, record, config, torch.device('cpu'), 1)
            self.assertTrue(info['local_valid'])
            self.assertGreater(value, 0)
            self.assertEqual(graph.last_stats['response_count'], 1)
            self.assertEqual(graph.last_stats['tail_silence_seconds'], 0)

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    def test_padding_uses_normal_forward_but_has_zero_gradients(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            config, graph, record, clip = self.setup_graph(directory)
            graph.padding_record = record
            engine = Engine(graph)
            def reduce(tensor, **kwargs):
                if tensor.dtype == torch.int64:
                    tensor.fill_(1)
            with patch('torch.distributed.all_reduce', side_effect=reduce), \
                 patch('train.src.training.data_parallel.Clip', return_value=clip):
                _, result = step(engine, None, config, torch.device('cpu'), 2)
            self.assertFalse(result['local_valid'])
            self.assertEqual(engine.calls, 1)
            self.assertEqual(graph.last_stats['silence_tokens'], 0)
            self.assertEqual(graph.last_stats['response_tokens'], 0)
            self.assertIsNone(graph.last_stats['loss_response'])
            for p in graph.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(p.grad.abs().sum().item(), 0.)

    @patch('train.src.models.streaming.user_request', return_value='question')
    @patch('train.src.models.streaming.system_prompt', return_value='system')
    @patch('torch.distributed.all_reduce')
    def test_all_padding_advances_backward_and_reports_effective_batch(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            config, graph, record, clip = self.setup_graph(directory)
            config['gradient_accumulation_steps'] = 4
            graph.padding_record = record
            engine = Engine(graph)
            with patch('train.src.training.data_parallel.Clip', return_value=clip):
                value, info = step(engine, None, config, torch.device('cpu'), 4)
            self.assertEqual(value, 0.)
            self.assertEqual(info['valid_qa'], 0)
            self.assertEqual(info['global_batch_size'], 16)
            self.assertEqual(engine.calls, 1)
            for p in graph.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    self.assertEqual(p.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
