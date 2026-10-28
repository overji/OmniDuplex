"""Automatic evaluation must not treat a stopped run or partial save as trained."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train.scripts.benchmark_after_training import completed_training, external_weights, prepare, sha


class AfterTrainingTests(unittest.TestCase):
    def test_weight_storage_rejects_repo_and_all_home_jimodels_children(self):
        with self.assertRaises(ValueError):
            external_weights(Path.home() / 'jimodels/another_project/weights')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(external_weights(directory), Path(directory).resolve())

    def test_complete_run_required_on_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {'world_size': 2, 'config': {'epochs': 2}}
            (root / 'run_config.json').write_text(json.dumps(run))
            config = {'training_output': directory, 'training_run_sha256': sha(root / 'run_config.json'),
                      'expected_optimizer_steps': 500}
            with self.assertRaisesRegex(ValueError, 'without final report'):
                completed_training(config)
            for rank in range(2):
                (root / f'memory_report.rank{rank}.json').write_text(json.dumps(
                    {'checkpoint_saved': True, 'optimizer_steps': 17}))
            with self.assertRaisesRegex(ValueError, 'stopped early'):
                completed_training(config)
            for rank in range(2):
                (root / f'memory_report.rank{rank}.json').write_text(json.dumps(
                    {'checkpoint_saved': True, 'optimizer_steps': 500}))
            checkpoint = root / 'checkpoint-final'; (checkpoint / 'state').mkdir(parents=True)
            (checkpoint / 'training_config.json').write_text(json.dumps(run['config']))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                completed_training(config)
            for rank in range(2):
                (checkpoint / f'rng_rank{rank}.pt').write_bytes(b'saved')
                (checkpoint / 'state' / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'saved')
            (checkpoint / 'state/mp_rank_00_model_states.pt').write_bytes(b'saved')
            self.assertEqual(completed_training(config), checkpoint)
            (root / 'run_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                completed_training(config)

    @patch('train.scripts.benchmark_after_training.subprocess.run')
    def test_plan_only_has_realtime_and_no_reused_baseline_predictions(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.json'; data.write_text('[]')
            config = {'output': str(root / 'results'), 'data': str(data), 'data_sha256': sha(data),
                      'training_output': str(root / 'training'), 'export_model': str(root / 'weights'),
                      'benchmark_python': '/python', 'videos_dir': str(root), 'gpus': [0, 1, 2, 3]}
            suite = prepare(config)
            commands = json.loads((root / 'results/commands.json').read_text())
            self.assertEqual([c['mode'] for c in commands], ['realtime'])
            self.assertNotIn('--reuse-non-agent', commands[0]['argv'])
            self.assertNotIn('--reuse-root', suite)
            self.assertEqual(suite[-4:], ['0', '1', '2', '3'])
            self.assertEqual(run.call_args.args[0][-1], '--dry-run')


if __name__ == '__main__':
    unittest.main()
