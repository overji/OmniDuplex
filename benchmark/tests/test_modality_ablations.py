import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

from benchmark.adapters.minicpmo_ablation import filter_prefill
from benchmark.cli.main import build_parser
from benchmark.probe import models
from benchmark.probe.prompts import ProbeInput
from benchmark.scripts.resume_after_agent import scheduled_commands
from benchmark.scripts.run_prepared_suite import build_plan


class ModalityTests(unittest.TestCase):
    def test_minicpm_omits_removed_modality_and_keeps_text_and_options(self):
        audio, frames = object(), [object()]
        for modality, removed, retained in [('video-only', 'audio_waveform', 'frame_list'),
                                             ('audio-only', 'frame_list', 'audio_waveform')]:
            prefill = Mock(return_value={'success': True})
            filtered = filter_prefill(prefill, modality)
            filtered(audio_waveform=audio, frame_list=frames, max_slice_nums=1)
            self.assertNotIn(removed, prefill.call_args.kwargs)
            self.assertIn(retained, prefill.call_args.kwargs)
            self.assertEqual(prefill.call_args.kwargs['max_slice_nums'], 1)
            filtered(text_list=['standing question'])
            self.assertEqual(prefill.call_args.kwargs, {'text_list': ['standing question']})

    def test_qwen_ablations_omit_inputs_at_processor_and_model(self):
        class Features(dict):
            def to(self, *args):
                return self
        for modality in ('video-only', 'audio-only'):
            args = NS(probe_model='qwen25_omni', input_modality=modality,
                      fps=1, max_pixels=1024, max_new_tokens=16)
            probe = ProbeInput('system', 'question', Path('prefix.mp4'), True, 3.)
            sequence = MagicMock()
            model = NS(device='cpu', dtype='bfloat16', config=NS(), generate=Mock(return_value=sequence))
            processor = Mock(return_value=Features(input_ids=NS(shape=(1, 3))))
            processor.batch_decode.return_value = ['NO']
            media = ([object()], None, None) if modality == 'audio-only' else (None, None, [[1, 2, 3]])
            state = NS(model=model, processor=processor, process_mm_info=Mock(return_value=media),
                       torch=NS(inference_mode=contextlib.nullcontext))
            self.assertEqual(models.generate(args, state, probe), 'NO')
            content = processor.apply_chat_template.call_args.args[0][1]['content']
            self.assertEqual(content[0]['type'], 'audio' if modality == 'audio-only' else 'video')
            self.assertFalse(model.generate.call_args.kwargs['use_audio_in_video'])
            kwargs = processor.call_args.kwargs
            self.assertIsNone(kwargs['videos'] if modality == 'audio-only' else kwargs['audio'])

    def test_audio_only_missing_track_is_explicit(self):
        args = NS(input_modality='audio-only')
        probe = ProbeInput('system', 'question', Path('prefix.mp4'), False, 3.)
        with self.assertRaisesRegex(ValueError, 'requires a source audio'):
            models.conversation(probe, args)

    def test_ablation_options_are_opt_in_and_survive_parallel_cli(self):
        parser = build_parser()
        argv = ['parallel', 'minicpmo', '--model-path', '/model', '--gpus', '0', '1']
        self.assertFalse(hasattr(parser.parse_args(argv), 'input_modality'))
        self.assertEqual(parser.parse_args(argv + ['--input-modality', 'video-only']).input_modality, 'video-only')

    def test_phased_schedule_gpus_order_and_disjoint_outputs(self):
        names = ['minicpmo', 'qwen3_omni', 'qwen25_omni', 'ming_omni', 'qwen3_vl',
                 'baichuan_omni', 'humanomni_v2', 'omnivinci']
        entries = [dict(mode='probe', adapter=n, argv=['python', '-m', 'benchmark', 'probe', 'run', n,
                                                     '--manifest', '/manifest.jsonl']) for n in names]
        entries += [dict(mode='realtime', adapter=n, argv=['python', '-m', 'benchmark', 'run', n])
                    for n in ['minicpmo', 'gander', 'livestar', 'joyai']]
        remaining = [e for e in entries if e['mode'] == 'realtime' and e['adapter'] != 'minicpmo']
        path = Path(__file__).resolve().parents[1] / 'configs/suites/post_realtime_20261023.json'
        schedule = json.loads(path.read_text())
        gpus = ['0', '1', '2', '3', '6', '7']
        commands = scheduled_commands(entries, remaining, schedule, gpus, 'task-start')
        self.assertEqual([e['adapter'] for e in commands[:3]], ['gander', 'livestar', 'joyai'])
        self.assertEqual([e['adapter'] for e in commands[3:6]], names[:3])
        self.assertEqual([e['variant'] for e in commands[6:10]], ['no_audio', 'no_vision'] * 2)
        self.assertTrue(all(e['gpus'] == ['0', '1'] for e in commands[6:]))
        self.assertTrue(all(e['gpus'] == gpus for e in commands[:6]))
        rows = [{'pair_id': str(i), 'probe_id': f'{i}:{s}'} for i in range(6) for s in ('pre', 'post')]
        with tempfile.TemporaryDirectory() as tmp, patch('benchmark.scripts.run_prepared_suite.load_manifest', return_value=rows):
            tasks = build_plan(commands, gpus, Path(tmp))
            self.assertEqual(len({t['output'] for t in tasks}), len(tasks))
            self.assertEqual([t['adapter'] for t in tasks], [c['adapter'] for c in commands])
            self.assertEqual(len(tasks[3]['commands']), 6)
            self.assertEqual(len(tasks[4]['commands']), 3)
            self.assertEqual(tasks[4]['commands'][0]['gpus'], ['0', '1'])
            self.assertEqual(len(tasks[5]['commands']), 6)
            for i in (6, 7):
                cmd = tasks[i]['commands'][0]
                j = cmd.index('--gpus')
                self.assertEqual(cmd[j + 1:j + 3], ['0', '1'])
            self.assertTrue(all(len(tasks[i]['commands']) == 2 for i in range(8, len(tasks))))
            eight = [str(i) for i in range(8)]
            schedule8 = json.loads(path.with_name('post_realtime_8gpu_20261023.json').read_text())
            commands8 = scheduled_commands(entries, remaining, schedule8, eight, 'task-start')
            self.assertEqual([e['adapter'] for e in commands8[:3]], ['gander', 'joyai', 'livestar'])
            self.assertTrue(all(e['gpus'] == eight for e in commands8))
            tasks8 = build_plan(commands8, eight, Path(tmp) / 'eight')
            self.assertEqual(len(tasks8[4]['commands']), 4)
            for task in tasks8:
                if task['mode'] == 'probe' and task['adapter'] != 'qwen3_omni':
                    self.assertEqual(len(task['commands']), 8)
                elif task['mode'] == 'realtime':
                    command = task['commands'][0]
                    start = command.index('--gpus')
                    self.assertEqual(command[start + 1:start + 9], eight)


if __name__ == '__main__':
    unittest.main()
