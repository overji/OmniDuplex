import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmark.probe.runner import pending_requests
from benchmark.scripts import run_prepared_suite
from benchmark.tests.test_probe import probes


class ProbeRetryTests(unittest.TestCase):
    def test_unattempted_first_then_one_retry_and_successes_preserved(self):
        rows = [{'probe_id': key} for key in ('done', 'old', 'new')]
        records = [{'probe_id': 'done'}]
        errors = [{'probe_id': 'old'}] * 7
        baseline = {'old': 7}
        generated = []
        for row in pending_requests(rows, records, errors, baseline):
            generated.append(row['probe_id'])
            errors.append({'probe_id': row['probe_id']})
        self.assertEqual(generated, ['new', 'old', 'new'])
        self.assertEqual(list(pending_requests(rows, records, errors, baseline)), [])

    def test_restart_does_not_replenish_budget_and_success_is_not_retried(self):
        rows = [{'probe_id': key} for key in ('old', 'new')]
        records = []
        errors = [{'probe_id': 'old'}] * 3
        baseline = {'old': 3}
        pending = pending_requests(rows, records, errors, baseline)
        self.assertEqual(next(pending)['probe_id'], 'new')
        errors.append({'probe_id': 'new'})
        # Restart after first-pass failure, with the original persistent baseline.
        attempted = []
        for row in pending_requests(rows, records, errors, baseline):
            attempted.append(row['probe_id'])
            records.append({'probe_id': row['probe_id']})
        self.assertEqual(attempted, ['old', 'new'])
        self.assertEqual(list(pending_requests(rows, records, errors, baseline)), [])

    def test_legacy_keeps_single_pending_pass(self):
        rows = [{'probe_id': key} for key in ('done', 'failed')]
        self.assertEqual(list(pending_requests(rows, [{'probe_id': 'done'}],
                                              [{'probe_id': 'failed'}])), [rows[1]])

    def test_suite_continues_to_next_model_after_failed_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data.json';data.write_text('[]')
            commands = root/'commands.json'
            commands.write_text(json.dumps([{'argv': ['unused','--data',str(data),
                                                      '--videos-dir',str(root)]}]))
            manifest = root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in probes()[0]))
            tasks = [dict(mode='probe',adapter=name,smoke=None,manifest=str(manifest),
                          output=str(root/name),commands=[dict(gpus=['0'],
                          argv=[sys.executable,'-c','raise SystemExit(1)'])])
                     for name in ('first','second')]
            argv = ['suite','--commands',str(commands),'--output-dir',str(root/'suite'),
                    '--gpus','0','--probe-retry-errors-once']
            with patch.object(sys,'argv',argv), \
                 patch.object(run_prepared_suite,'build_plan',return_value=tasks), \
                 patch.object(run_prepared_suite.time,'sleep'):
                self.assertEqual(run_prepared_suite.main(),0)
            status = json.loads((root/'suite/suite_status.json').read_text())
            self.assertEqual(status['status'],'complete_with_errors')
            self.assertEqual([job['adapter'] for job in status['jobs']],['first','second'])
            self.assertTrue(all(job['status']=='incomplete_final' for job in status['jobs']))
