import contextlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark.data.files import read_jsonl, recover_jsonl_tail
from benchmark.data.media import media_cache
from benchmark.evaluation.timing import summarize
from benchmark.runtime import parallel


FAKE_WORKER = '''
import json, os, pathlib, sys, time
def flag(name):
    return sys.argv[sys.argv.index('--' + name) + 1]
output = pathlib.Path(flag('output-dir'))
rank = int(flag('internal-rank'))
plan = json.loads(pathlib.Path(flag('internal-shard-plan')).read_text())
samples = json.loads(pathlib.Path(flag('data')).read_text())
(output / 'started.json').write_text(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'pid': os.getpid()}))
if rank == plan['arguments'].get('fail_rank', -1):
    print('intentional worker failure', flush=True)
    sys.exit(7)
path = output / 'predictions.jsonl'
done = {json.loads(line)['sample_index'] for line in path.read_text().splitlines()} if path.exists() else set()
for index in plan['assignments'][rank]:
    if index in done:
        continue
    sample = samples[index]
    prediction = {'sample_index': index, 'task': sample['task'], 'video': sample['video'],
                  'question': sample['question'], 'status': 'complete',
                  'events': [{'onset_seconds': 2.0, 'text': 'reply'}]}
    with path.open('a') as f:
        f.write(json.dumps(prediction) + '\\n')
        f.flush()
    time.sleep(0.02)
'''


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.samples = [dict(task=str(i), video='shared.mp4', question='Request', question_time='00:01',
                             mode='pr', ability='trigger', responses=[dict(
                                 window_start='00:01', window_end='00:03')]) for i in range(7)]
        (root / 'data.json').write_text(json.dumps(self.samples))
        (root / 'prompt.txt').write_text('Prompt')
        self.script = root / 'fake_worker.py'
        self.script.write_text(FAKE_WORKER)
        self.args = SimpleNamespace(data=root / 'data.json', system_prompt=root / 'prompt.txt',
                                    model_path=root / 'model', output_dir=root / 'result',
                                    max_samples=0)

    def run_workers(self, **controls):
        original = parallel.worker_command
        def command(*args, **kwargs):
            native = original(*args, **kwargs)
            return [sys.executable, str(self.script), *native[4:]]
        with patch.object(parallel, 'worker_command', side_effect=command), \
                contextlib.redirect_stdout(io.StringIO()):
            return parallel.run_parallel(self.args, 'fake', ['4', '5', '6', '7'],
                                         poll_seconds=0.01, **controls)

    def test_selected_original_indices_keep_identity_and_balance_work(self):
        self.assertEqual(parallel.assignments(self.samples, self.args, 4, [1, 2, 4, 5, 6]),
                         [[1, 6], [2], [4], [5]])
        for invalid in ([1, 1], [-1], [7], [True], {"indices": [1]}):
            with self.assertRaisesRegex(ValueError, "sample-indices"):
                parallel.assignments(self.samples, self.args, 4, invalid)
        self.assertEqual(parallel.assignments(self.samples, self.args, 4, []), [[], [], [], []])

    def test_moss_restarts_only_failed_shard_and_keeps_terminal_failures(self):
        script = FAKE_WORKER.replace("'status': 'complete'", "'status': 'timeout' if not done else 'complete'")
        script = script.replace('    time.sleep(0.02)', '    if not done:\n        sys.exit(75)\n    time.sleep(0.02)')
        self.script.write_text(script)
        original = parallel.worker_command
        def command(*args, **kwargs):
            native = original(*args, **kwargs)
            return [sys.executable, str(self.script), *native[4:]]
        with patch.object(parallel, 'worker_command', side_effect=command), \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                self.assertEqual(parallel.run_parallel(self.args, 'moss_vl', ['6', '7'],
                                                      poll_seconds=.01), 0)
        rows = read_jsonl(self.args.output_dir / 'predictions.jsonl')
        self.assertEqual(len(rows), len(self.samples))
        self.assertEqual(len({r['sample_index'] for r in rows}), len(self.samples))
        self.assertEqual(sum(r['status'] == 'timeout' for r in rows), 2)
        status = json.loads((self.args.output_dir / 'parallel_status.json').read_text())
        self.assertEqual(status['status'], 'complete_with_errors')

    def test_selected_indices_run_without_renumbering_and_resume(self):
        selection = Path(self.temp.name) / 'indices.json'
        selection.write_text(json.dumps([1, 2, 4, 5, 6]))
        self.assertEqual(self.run_workers(sample_indices=selection), 0)
        predictions = read_jsonl(self.args.output_dir / 'predictions.jsonl')
        self.assertEqual([row['sample_index'] for row in predictions], [1, 2, 4, 5, 6])
        state = json.loads((self.args.output_dir / 'parallel_status.json').read_text())
        self.assertEqual(state['expected_samples'], 5)
        self.assertEqual(state['status'], 'complete')
        before = (self.args.output_dir / 'predictions.jsonl').read_bytes()
        self.assertEqual(self.run_workers(sample_indices=selection), 0)
        self.assertEqual((self.args.output_dir / 'predictions.jsonl').read_bytes(), before)

    def test_fixed_partition_limits_and_shards(self):
        shards = parallel.assignments(self.samples, self.args, 4)
        self.assertEqual(shards, [[0, 4], [1, 5], [2, 6], [3]])
        self.args.max_samples = 3
        self.assertEqual(parallel.assignments(self.samples, self.args, 4), [[0], [1], [2], []])
        self.args.max_samples = 0
        self.args.sample_start, self.args.sample_step = 1, 2
        self.assertEqual(parallel.assignments(self.samples, self.args, 2), [[1, 5], [3]])

    def test_real_process_isolation_live_collection_resume_and_score_equivalence(self):
        self.assertEqual(self.run_workers(), 0)
        root = self.args.output_dir
        for rank, gpu in enumerate(['4', '5', '6', '7']):
            started = json.loads((root / f'worker_{rank}/started.json').read_text())
            self.assertEqual(started['gpu'], gpu)
            self.assertNotEqual(started['pid'], os.getpid())
            with self.assertRaises(ProcessLookupError):
                os.kill(started['pid'], 0)
        predictions = read_jsonl(root / 'predictions.jsonl')
        self.assertEqual([p['sample_index'] for p in predictions], list(range(7)))
        self.assertEqual(json.loads((root / 'metrics.json').read_text()),
                         summarize(self.samples, predictions))
        before = (root / 'predictions.jsonl').read_bytes()
        self.assertEqual(self.run_workers(), 0)
        self.assertEqual((root / 'predictions.jsonl').read_bytes(), before)
        self.args.max_samples = 1
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            self.run_workers()

    def test_one_worker_failure_does_not_stop_other_shards(self):
        self.args.fail_rank = 1
        self.assertEqual(self.run_workers(), 1)
        root = self.args.output_dir
        state = json.loads((root / 'parallel_status.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['missing_sample_indices'], [1, 5])
        self.assertEqual(state['workers'][1]['exit_code'], 7)
        self.assertIn('intentional worker failure', state['workers'][1]['failure_tail'])
        self.assertEqual([p['sample_index'] for p in read_jsonl(root / 'predictions.jsonl')],
                         [0, 2, 3, 4, 6])

    def test_interrupted_supervisor_reaps_all_its_workers(self):
        spawn = parallel.subprocess.Popen
        owned = []
        def capture(*args, **kwargs):
            process = spawn(*args, **kwargs)
            owned.append(process)
            return process
        with patch.object(parallel.subprocess, 'Popen', side_effect=capture), \
                patch.object(parallel, 'collect', side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_workers(), 130)
        self.assertEqual(len(owned), 4)
        self.assertTrue(all(process.poll() is not None for process in owned))
        state = json.loads((self.args.output_dir / 'parallel_status.json').read_text())
        self.assertEqual(state['status'], 'interrupted')

    def test_unfinished_jsonl_record_is_ignored_and_recovered(self):
        path = self.args.data.parent / 'partial.jsonl'
        path.write_bytes(b'{"sample_index": 0}\n{"text":"\xe4')
        self.assertEqual(read_jsonl(path, complete_only=True), [{'sample_index': 0}])
        with contextlib.redirect_stdout(io.StringIO()):
            recover_jsonl_tail(path)
        self.assertEqual(path.read_bytes(), b'{"sample_index": 0}\n')
        path.write_text('{invalid}\n')
        with self.assertRaises(json.JSONDecodeError):
            read_jsonl(path, complete_only=True)

    def test_duplicate_gpus_are_rejected(self):
        self.args.output_dir.mkdir()
        with self.assertRaises(ValueError):
            parallel.prepare_plan(self.args, 'fake', ['4', '4'], sys.executable)

    def test_existing_serial_results_are_never_overwritten(self):
        self.args.output_dir.mkdir()
        path = self.args.output_dir / 'predictions.jsonl'
        path.write_text('{"sample_index": 0}\n')
        with self.assertRaisesRegex(ValueError, 'existing results'):
            self.run_workers()
        self.assertEqual(path.read_text(), '{"sample_index": 0}\n')

    def test_shared_video_cache_is_built_once_across_processes(self):
        cache = self.args.data.parent / 'cache'
        cache.mkdir()
        marker = cache / 'built'
        def prepare(video, directory):
            if not marker.exists():
                time.sleep(0.1)
                with (cache / 'builds').open('a') as f:
                    f.write('built\n')
                marker.touch()
        with patch('benchmark.data.media._media_cache_unlocked', side_effect=prepare):
            context = multiprocessing.get_context('fork')
            processes = [context.Process(target=media_cache,
                                         args=(cache / 'shared.mp4', cache)) for _ in range(4)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=3)
                self.assertEqual(process.exitcode, 0)
        self.assertEqual((cache / 'builds').read_text(), 'built\n')


if __name__ == '__main__':
    unittest.main()
