import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from train.src.infrastructure.tracking import start_tracking


class TrackingTests(unittest.TestCase):
    def test_only_rank_zero_uploads_metrics_without_sensitive_fields(self):
        run = MagicMock(id='test', url='https://example.test/run', project='project', entity='team')
        sdk = SimpleNamespace(init=MagicMock(return_value=run))
        config = dict(wandb={'project': 'project'}, learning_rate=1e-6, api_key='private', data='private')
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, wandb=sdk):
            output = Path(directory)
            start_tracking(config, output, 1, 4).log_train({'step': 1, 'loss': 2.})
            sdk.init.assert_not_called()
            tracker = start_tracking(config, output, 0, 4)
            self.assertEqual(sdk.init.call_args.kwargs['config'], {'learning_rate': 1e-6, 'world_size': 4, 'effective_batch_size': 1})
            self.assertEqual(json.loads((output/'wandb_run.json').read_text())['id'], 'test')
            tracker.log_train({'step': 1, 'loss': 2., 'targets': ['private'], 'response_text': 'private'})
            self.assertEqual(run.log.call_args.args[0], {'train/step': 1, 'train/loss': 2.})
            tracker.log_validation(1, 1.5, 3, 1)
            self.assertEqual(run.log.call_args.args[0]['validation/loss'], 1.5)
            tracker.finish({'optimizer_steps': 1})
            run.finish.assert_called_once()

    def test_disabled_tracking_does_not_initialize_sdk(self):
        sdk = SimpleNamespace(init=MagicMock())
        with patch.dict(sys.modules, wandb=sdk):
            for config in ({}, {'wandb': {'mode': 'disabled'}}):
                tracker = start_tracking(config, Path('/unused'), 0, 4)
                tracker.log_train({})
                tracker.finish({})
        sdk.init.assert_not_called()


if __name__ == '__main__':
    unittest.main()
