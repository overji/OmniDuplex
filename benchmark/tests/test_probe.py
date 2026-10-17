"""Offline probe contract tests. This feature does not require model execution to score."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

from benchmark.cli.main import build_parser
from benchmark.probe import data, media, metrics, models, runner
from benchmark.probe.prompts import model_input


def sample():
    return {'task': 'object_perception_pr', 'mode': 'pr', 'ability': 'object_perception',
        'question': 'Tell me whenever a cup moves.', 'question_time': '00:00',
        'available_tools': [], 'responses': [
            {'window_start': '00:10', 'window_end': '00:12', 'response': 'SECRET REFERENCE',
             'evidence': [{'fact': 'SECRET EVIDENCE'}]},
            {'window_start': '00:20', 'window_end': '00:22'}]}


def probes(qa=None):
    return data.sample_probes(qa or sample(), 0, 'source-hash', Path('/video.mp4'),
        {'duration': 30, 'has_audio': True, 'source': {'path': '/video.mp4', 'size': 1, 'mtime_ns': 1}})


def prediction(row, raw):
    return {'probe_id': row['probe_id'], 'status': 'complete', 'raw_response': raw}


class ProbeDataTests(unittest.TestCase):
    def test_cutoffs_pairs_and_reference_free_model_input(self):
        rows, exclusions = probes()
        self.assertEqual(exclusions, [])
        self.assertEqual([r['prefix_end_seconds'] for r in rows], [9., 12., 19., 22.])
        self.assertEqual([r['handled_count'] for r in rows], [0, 0, 1, 1])
        self.assertEqual([r['expected'] for r in rows], ['NO', 'YES', 'NO', 'YES'])
        self.assertEqual(rows[0]['pair_id'], rows[1]['pair_id'])
        inputs = model_input(rows[0], Path('/prefix.mp4'), 'Probe rules')
        joined = inputs.system_prompt + inputs.user_prompt
        self.assertIn(sample()['question'], joined)
        self.assertNotIn('SECRET', joined)
        self.assertNotIn('window', inputs.user_prompt)
        self.assertNotIn('00:10', inputs.user_prompt)
        changed = copy.deepcopy(rows[0])
        changed.update(side='post', expected='YES', window_start_seconds=999)
        changed['request']['responses'] = [{'response': 'ANOTHER SECRET'}]
        self.assertEqual(model_input(changed, inputs.video, 'Probe rules'), inputs)

    def test_excluded_first_pair_still_counts_as_handled_for_second(self):
        qa = sample()
        qa['responses'][0].update(window_start='00:01', window_end='00:02')
        rows, exclusions = probes(qa)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['handled_count'], 1)
        self.assertEqual(exclusions[0]['reason'], 'negative_prefix_not_after_question')

    def test_close_occurrences_cannot_claim_an_unseen_previous_event(self):
        qa = sample()
        qa['responses'][1].update(window_start='00:11', window_end='00:12')
        rows, exclusions = probes(qa)
        self.assertEqual(len(rows), 2)
        self.assertEqual(exclusions[0]['reason'], 'prior_opportunity_not_observed_before_negative_cutoff')

    def test_agent_results_and_completion_are_not_probe_targets(self):
        qa = sample()
        qa.update(mode='agent', task='object_perception_agent', sample_type='positive')
        qa['responses'][0].update(role='assistant', tool_calls=[{'function': {'name': 'agent_call'}}])
        qa['responses'][1]['role'] = 'tool'
        qa['responses'].append(dict(qa['responses'][1], role='assistant'))
        rows, _ = probes(qa)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['response_index'], 0)

    def test_out_of_video_windows_and_incomplete_manifest_are_rejected(self):
        qa = sample()
        qa['responses'][1]['window_end'] = '00:31'
        with self.assertRaises(ValueError):
            probes(qa)
        rows, _ = probes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.jsonl'
            path.write_text(json.dumps(rows[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'Every manifest pair'):
                data.load_manifest(path)

    def test_clipping_filters_both_modalities_before_cutoff_and_omits_absent_audio(self):
        command = media.prefix_command(Path('video.mp4'), Path('prefix.mp4'), 9, True)
        self.assertIn('trim=start=0:end=9.000000000', command)
        self.assertIn('atrim=start=0:end=9.000000000', command)
        self.assertIn('-copyts', command)
        self.assertNotIn('copy', command)
        self.assertNotIn('0:a:0', media.prefix_command(Path('v.mp4'), Path('p.mp4'), 9, False))
        self.assertIn('-an', media.prefix_command(Path('v.mp4'), Path('p.mp4'), 9, False))

    def test_prefix_validation_rejects_a_frame_exactly_at_the_cutoff(self):
        with patch.object(media, 'scan_video', return_value={'duration': 9., 'has_audio': False}), \
                patch.object(media.subprocess, 'run', return_value=NS(stdout=json.dumps(
                    {'frames': [{'best_effort_timestamp_time': '9.000'}]}))):
            with self.assertRaisesRegex(ValueError, 'cross the cutoff'):
                media.validate_prefix(Path('prefix.mp4'), 9., False)

    def test_changed_source_cannot_reuse_prefix_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            video.write_bytes(b'old')
            identity = media.source_identity(video)
            video.write_bytes(b'new source')
            with self.assertRaisesRegex(ValueError, 'changed'):
                media.prefix_clip({'video': str(video), 'source': identity}, root / 'cache')


class ProbeMetricsTests(unittest.TestCase):
    def test_invalid_is_never_a_correct_negative(self):
        rows, _ = probes()
        records = [prediction(rows[0], 'The answer is NO'), prediction(rows[1], 'YES: A cup moved.')]
        score = metrics.summarize(rows, records)['groups']['overall']
        self.assertEqual(score['accuracy'], .5)
        self.assertEqual(score['negative_accuracy'], 0)
        self.assertEqual(score['paired_accuracy'], 0)
        self.assertEqual(score['parse_coverage'], .5)
        self.assertEqual(score['pair_coverage'], .5)
        self.assertEqual(score['valid_only_precision'], 1)
        for text in ('', 'YES or NO', '```NO```', 'I think YES', 'YES:'):
            self.assertIsNone(metrics.parse_decision(text))
        self.assertEqual(metrics.parse_decision('no.'), 'NO')
        self.assertEqual(metrics.parse_decision('YES: New movement.'), 'YES')

    def test_failed_and_partial_pairs_are_coverage_not_false_negative(self):
        rows, _ = probes()
        records = [prediction(rows[0], 'NO')]
        result = metrics.summarize(rows, records, [{'probe_id': rows[1]['probe_id']}])
        overall = result['groups']['overall']
        self.assertEqual(overall['accuracy'], 1)
        self.assertEqual(overall['probe_coverage'], .25)
        self.assertIsNone(overall['paired_accuracy'])
        self.assertEqual(result['failed_probes'], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            metrics.summarize(rows, records + records)


class ProbeAdapterTests(unittest.TestCase):
    def test_text_only_generation_uses_fresh_conversations_and_handles_both_outputs(self):
        rows, _ = probes()
        args = NS(fps=1, max_pixels=151200, max_new_tokens=256)
        # A real BatchFeature is a dict. Keep tensor operations mocked for this contract test.
        class Features(dict):
            def to(self, *unused):
                return self
        features = Features(input_ids=NS(shape=(1, 10)))
        generated = MagicMock(spec=['__getitem__'])
        generated.__getitem__.return_value = 'new_tokens'
        model = NS(device='cuda:0', dtype='bfloat16', config=NS(), generate=Mock(return_value=(NS(sequences=generated), None)))
        processor = Mock(return_value=features)
        processor.batch_decode.return_value = ['YES: Movement.']
        import contextlib
        state = NS(model=model, processor=processor,
            process_mm_info=Mock(return_value=(None, None, [list(range(12))])),
            torch=NS(inference_mode=contextlib.nullcontext))
        probe = model_input(rows[1], Path('prefix.mp4'), 'Probe rules')
        self.assertEqual(models.generate(args, state, probe), 'YES: Movement.')
        self.assertFalse(model.generate.call_args.kwargs['return_audio'])
        self.assertFalse(model.generate.call_args.kwargs['thinker_do_sample'])
        self.assertEqual(model.generate.call_args.kwargs['thinker_max_new_tokens'], 256)
        self.assertEqual(generated.__getitem__.call_args.args[0][1], slice(10, None))
        first_messages = processor.apply_chat_template.call_args.args[0]
        model.generate.return_value = generated
        from dataclasses import replace
        silent = replace(probe, has_audio=False)
        models.generate(args, state, silent)
        self.assertIsNot(first_messages, processor.apply_chat_template.call_args.args[0])
        self.assertFalse(state.process_mm_info.call_args.kwargs['use_audio_in_video'])
        model.config.max_position_embeddings = 20
        with self.assertRaisesRegex(ValueError, 'exceeds model context'):
            models.generate(args, state, probe)
        self.assertEqual(model.generate.call_count, 2)

    def test_cli_and_resume_retry_only_the_failed_probe(self):
        parser = build_parser()
        for name in ('qwen3_omni', 'qwen25_omni'):
            args = parser.parse_args(['probe', 'run', name, '--manifest', '/manifest',
                                     '--model-path', '/weights', '--output-dir', '/out'])
            self.assertEqual(args.max_new_tokens, 256)
            self.assertEqual(args.fps, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.jsonl'
            rows, _ = probes()
            rows = rows[:2]
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'config.json').write_text('{}')
            prompt = root / 'prompt.txt'
            prompt.write_text('Probe rules')
            args = parser.parse_args(['probe', 'run', 'qwen25_omni', '--manifest', str(manifest),
                '--model-path', str(model_dir), '--system-prompt', str(prompt), '--output-dir', str(root / 'out')])
            with patch('benchmark.probe.models.load'), patch.object(runner, 'prefix_clip', return_value=root / 'prefix.mp4'), \
                    patch('benchmark.probe.models.generate', side_effect=['NO', RuntimeError('OOM')]):
                self.assertEqual(runner.run(args), 1)
            with (args.output_dir / 'probe_predictions.jsonl').open('a') as unfinished:
                unfinished.write('{"probe_id":')
            with patch('benchmark.probe.models.load'), patch.object(runner, 'prefix_clip', return_value=root / 'prefix.mp4'), \
                    patch('benchmark.probe.models.generate', return_value='YES: Movement.') as generate:
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(generate.call_count, 1)
            metrics_json = json.loads((args.output_dir / 'probe_metrics.json').read_text())
            self.assertEqual(metrics_json['groups']['overall']['paired_accuracy'], 1)
            self.assertEqual(metrics_json['failed_probes'], 0)
            args.fps = 2
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                runner.run(args)
