"""A slow native sample must not block the following sample or a resume."""
import json
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from benchmark.adapters import livestar
from benchmark.data.types import SampleResult
from benchmark.runtime.runner import run


class LiveStarTimeoutTests(unittest.TestCase):
    def test_timeout_restores_hooks_continues_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [dict(task='test_pr', mode='pr', ability='event_understanding',
                            video='v.mp4', question='Notify', question_time='00:00',
                            responses=[]) for _ in range(2)]
            (root / 'data.json').write_text(json.dumps(samples))
            (root / 'prompt.txt').write_text('Observe.')
            args = NS(data=root / 'data.json', system_prompt=root / 'prompt.txt',
                      output_dir=root / 'out', max_samples=0, sample_timeout_seconds=.02)
            chat, generate = object(), object()
            state = NS(model=NS(chat=chat, generate=generate))
            calls = []

            def native(a, s, context):
                calls.append(context.index)
                context.build_prompt()
                self.assertIs(s.model.chat, chat)
                self.assertIs(s.model.generate, generate)
                if context.index == 0:
                    s.model.chat = object()
                    s.model.generate = object()
                    time.sleep(.2)
                return SampleResult(events=[], observed_seconds=3.)

            adapter = NS(run_config=lambda a, b: {}, load=lambda a: state,
                         run_sample=livestar.run_sample)
            handler = signal.getsignal(signal.SIGALRM)
            with patch.object(livestar, '_run_sample', side_effect=native):
                run(args, adapter)
                run(args, adapter)
            self.assertEqual(calls, [0, 1])
            rows = [json.loads(s) for s in (args.output_dir / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows], ['timeout', 'complete'])
            self.assertEqual(rows[0]['observed_seconds'], 0.)
            self.assertEqual(json.loads((args.output_dir / 'sample_000.json').read_text())['status'], 'timeout')
            self.assertIs(state.model.chat, chat)
            self.assertIs(state.model.generate, generate)
            self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.)

    def test_invalid_deadline_is_rejected(self):
        for limit in (-1., float('nan'), float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                livestar.run_config(NS(sample_timeout_seconds=limit), '')


if __name__ == '__main__':
    unittest.main()
