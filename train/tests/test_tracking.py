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

    def test_component_counts_nulls_and_validation_are_reported(self):
        from train.src.infrastructure.tracking import Tracking
        run = MagicMock()
        tracker = Tracking(run)
        tracker.log_train(dict(step=251, loss=1.9, loss_silence=.2, silence_tokens=30,
            silence_qa=4, loss_agent_instruction=None, agent_instruction_tokens=0,
            loss_response_contribution=1.1))
        values = run.log.call_args.args[0]
        self.assertEqual(values['train/loss_silence'], .2)
        self.assertEqual(values['train/silence_tokens'], 30)
        self.assertEqual(values['train/agent_instruction_tokens'], 0)
        self.assertNotIn('train/loss_agent_instruction', values)
        tracker.log_validation(250, 2., 12, 1, dict(loss_speak=.4, speak_tokens=20,
                                                  loss_agent_instruction=None))
        values = run.log.call_args.args[0]
        self.assertEqual(values['validation/loss_speak'], .4)
        self.assertNotIn('validation/loss_agent_instruction', values)

    def test_resumed_wandb_run_does_not_log_a_new_zero_step(self):
        run = MagicMock(id='existing', url='https://example.test/run', project='project', entity='team')
        run.resumed = True
        sdk = SimpleNamespace(init=MagicMock(return_value=run))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, wandb=sdk):
            start_tracking(dict(wandb={'project': 'project'}), Path(directory), 0, 4)
        run.log.assert_not_called()


if __name__ == '__main__':
    unittest.main()
