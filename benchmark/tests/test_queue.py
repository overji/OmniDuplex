import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from benchmark.runtime.queue import execute


class QueueTests(unittest.TestCase):
    def test_failure_is_recorded_and_next_model_still_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {'output_dir': str(root / 'results'), 'gpus': [4, 5, 6, 7],
                      'jobs': [dict(model=name, python='/worker/python', arguments=[])
                               for name in ('first', 'second')]}
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            processes = [SimpleNamespace(pid=1001, wait=lambda: 7),
                         SimpleNamespace(pid=1002, wait=lambda: 0)]
            spawn = Mock(side_effect=processes)
            with patch('benchmark.runtime.queue.subprocess.Popen', spawn), \
                    contextlib.redirect_stdout(io.StringIO()):
                execute(SimpleNamespace(config=path))
            self.assertEqual(spawn.call_count, 2)
            state = json.loads((root / 'results/queue_status.json').read_text())
            self.assertEqual(state['status'], 'incomplete')
            self.assertEqual([job['exit_code'] for job in state['jobs']], [7, 0])
            for call in spawn.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index('--gpus') + 1:command.index('--gpus') + 5],
                                 ['4', '5', '6', '7'])
                self.assertTrue(call.kwargs['start_new_session'])


if __name__ == '__main__':
    unittest.main()
