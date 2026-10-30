"""CPU contracts for consumption clocks, native turn parsing and MOSS deadlines."""
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from benchmark.adapters import moss_vl
from benchmark.cli.main import build_parser


def chunk(text, clock=5., question=True):
    return dict(text=text, onset_seconds=clock, question_consumed=question)


class MossTests(unittest.TestCase):
    def test_cli_and_native_settings(self):
        args = build_parser().parse_args(['parallel', 'moss_vl', '--model-path', '/model',
            '--gpus', '6', '7', '--agent-output-protocol', 'task-text'])
        self.assertEqual(args.sample_timeout_seconds, 1200.)
        self.assertEqual(args.max_tokens_per_second, 12)
        self.assertEqual(args.gpus, ['6', '7'])

    def test_clock_does_not_credit_queued_or_unforwarded_frame(self):
        clock = moss_vl.InputClock()
        frames = moss_vl.InputQueue(clock)
        prompts = moss_vl.InputQueue(clock, prompt=True)
        out = moss_vl.OutputQueue(clock, time.perf_counter())
        frames.put(('frame', 43.)); prompts.put('Q')
        out.put('a'); self.assertIsNone(out.get()['onset_seconds'])
        frames.get(); prompts.get()
        out.put('b'); self.assertIsNone(out.get()['onset_seconds'])
        clock.forwarded()
        frames.put(('future', 44.))
        out.put('c'); item = out.get()
        self.assertEqual(item['onset_seconds'], 43.)
        self.assertTrue(item['question_consumed'])

    def test_native_turns_merge_and_ignore_prequestion_output(self):
        rows = [chunk('early', 1., False), chunk('<|silence|>', 1., False),
                chunk('<|response|>There ', 43.), chunk('are two.', 44.),
                chunk('<|silence|>', 45.), chunk('<|response|>Now three.', 46.),
                chunk('<|im_end|>', 47.)]
        result = moss_vl.narration_events(rows)
        self.assertEqual(result, [dict(text='There are two.', onset_seconds=43.),
                                  dict(text='Now three.', onset_seconds=46.)])

    def test_split_control_tags_and_literal_task_instruction(self):
        rows = [chunk('<|res', 4.), chunk('ponse|>I will check.<|task_', 5.),
                chunk('start|>Search nearby pharmacies.', 8.), chunk('<|silence|>', 9.)]
        result = moss_vl.narration_events(rows, task_text=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['tool_calls'][0]['function']['arguments']['instruction'],
                         'Search nearby pharmacies.')
        self.assertEqual(result[0]['response_text'], 'I will check.')
        self.assertEqual(result[0]['onset_seconds'], 5.)
        self.assertEqual(result[0]['agent_call_onset_seconds'], 8.)

    def test_unfinished_agent_turn_is_truncated(self):
        result = moss_vl.narration_events([chunk('OK<|task_start|>Search')], task_text=True)
        self.assertEqual(result[0]['agent_parse_status'], 'truncated')

    def test_repeated_response_markers_are_continuations(self):
        result = moss_vl.narration_events([chunk('<|response|>There ', 1.),
            chunk('<|response|>are two buckets.', 2.), chunk('<|silence|>', 3.)])
        self.assertEqual(result, [dict(text='There are two buckets.', onset_seconds=1.)])

    def test_deadline_and_restart_only_after_persistence_hook(self):
        handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with moss_vl.deadline(.01):
                time.sleep(.2)
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0.)
        moss_vl.after_sample(None, NS(restart_required=False))
        with self.assertRaises(SystemExit) as exc:
            moss_vl.after_sample(None, NS(restart_required=True))
        self.assertEqual(exc.exception.code, 75)


if __name__ == '__main__':
    unittest.main()
