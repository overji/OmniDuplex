import os
from pathlib import Path
import tempfile
import unittest

from benchmark.scripts.resume_after_agent import process_identity, remaining_indices
from benchmark.scripts.run_prepared_suite import build_plan


class SuiteContinuationTests(unittest.TestCase):
    def test_pending_keeps_partial_excludes_agent_and_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            videos = Path(tmp)
            (videos / 'present.mp4').touch()
            samples = [dict(mode=mode, video=video) for mode, video in (
                ('rtd', 'present.mp4'), ('pr', 'present.mp4'),
                ('agent', 'present.mp4'), ('rtd', 'missing.mp4'))]
            records = [dict(sample_index=0, status='complete'), dict(sample_index=1, status='partial')]
            self.assertEqual(remaining_indices(samples, videos, records), [1])

    def test_suite_preserves_explicit_subset_for_full_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            entry = dict(mode='realtime', adapter='gander', argv=[
                'python', '-m', 'benchmark', 'run', 'gander', '--sample-indices', '/selected.json'])
            task, = build_plan([entry], ['0', '1'], output)
            for command in (task['commands'][0], task['smoke']):
                self.assertEqual(command.count('--sample-indices'), 1)
                self.assertEqual(command[command.index('--sample-indices') + 1], '/selected.json')

    def test_suite_default_still_uses_available_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = dict(mode='realtime', adapter='minicpmo', argv=['python', '-m', 'benchmark', 'run', 'minicpmo'])
            task, = build_plan([entry], ['0'], Path(tmp))
            command = task['commands'][0]
            self.assertEqual(command[command.index('--sample-indices') + 1], str(Path(tmp) / 'available_sample_indices.json'))

    def test_process_identity(self):
        self.assertIsNotNone(process_identity(os.getpid()))
        self.assertIsNone(process_identity(999999999))


if __name__ == '__main__':
    unittest.main()
