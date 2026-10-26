import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmark.scripts import watch_benchmark_queue as guard


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(output=str(self.root/'queue'), frozen_jobs=[dict(model='joyai',
            argv=['python','-m','benchmark','parallel','joyai','--output-dir',str(self.root/'joyai')])],
            next_argv=['python','suite.py','--output-dir',str(self.root/'suite'),'--gpus','0','1'],
            startup_grace_seconds=180, stale_after_seconds=3600, stop_grace_seconds=1,
            max_restarts_without_progress=3, repo=str(self.root))
        (self.root/'queue').mkdir(); (self.root/'joyai').mkdir()
        self.status = dict(pid=100,child_pid=101,status='running',stage='frozen_agent',
                           jobs=[dict(model='joyai',status='running')])
        guard.write(self.root/'queue/transition_status.json',self.status)

    def inspect(self):
        with patch.object(guard,'process',return_value=dict(pid=100)):
            return guard.inspect(self.config,1_000_000_000)

    def test_failed_worker_is_detected_even_if_supervisor_is_alive(self):
        guard.write(self.root/'joyai/parallel_status.json',dict(status='running',workers=[dict(rank=0,exit_code=1)]))
        self.assertEqual(self.inspect()['reason'],'realtime_worker_failed')

    def test_completed_workers_do_not_trigger_stall_repair(self):
        for i in range(2):
            directory=self.root/f'joyai/worker_{i}';directory.mkdir()
            (directory/'predictions.jsonl').write_text('{}\n')
        guard.write(self.root/'joyai/parallel_status.json',dict(status='running',workers=[
            dict(rank=0,exit_code=0),dict(rank=1,exit_code=None)]))
        snapshot=self.inspect();self.assertEqual(len(snapshot['workers']),1)
        self.assertTrue(snapshot['workers'][0]['key'].endswith('worker_1'))
        state={}
        self.assertEqual(guard.evaluate(snapshot,state,self.config,0)['verdict'],'healthy')
        self.assertEqual(guard.evaluate(snapshot,state,self.config,3599)['verdict'],'healthy')
        self.assertEqual(guard.evaluate(snapshot,state,self.config,3600)['reason'],'worker_no_progress')
        snapshot['workers'][0]['marker'][0]+=1
        self.assertEqual(guard.evaluate(snapshot,state,self.config,3601)['verdict'],'healthy')

    def test_real_chunk_output_counts_as_progress_partial_line_does_not(self):
        directory=self.root/'joyai'
        (directory/'predictions.jsonl').write_bytes(b'{}\n{"unfinished":')
        before=guard.marker(directory)
        (directory/'sample_1.chunks.jsonl').write_text('{}\n')
        after=guard.marker(directory)
        self.assertEqual(before[0],1);self.assertEqual(after[0],1)
        self.assertNotEqual(before[1],after[1])
        self.assertEqual(after[1],json.loads(json.dumps(after[1])))

    def test_manual_interrupt_is_respected_own_pending_repair_can_resume(self):
        guard.write(self.root/'queue/transition_status.json',{**self.status,'status':'interrupted'})
        snapshot=self.inspect();self.assertEqual(snapshot['verdict'],'paused')
        state={'pending_repair':dict(key='original',completed=17)}
        result=guard.evaluate(snapshot,state,self.config,0)
        self.assertEqual(result['verdict'],'repair');self.assertEqual(result['completed'],17)

    def test_complete_requires_complete_suite(self):
        guard.write(self.root/'queue/transition_status.json',{**self.status,'status':'complete'})
        (self.root/'suite').mkdir();guard.write(self.root/'suite/suite_status.json',dict(status='complete'))
        self.assertEqual(self.inspect()['verdict'],'complete')

    def test_reused_pid_is_never_signalled(self):
        with patch.object(guard,'process',return_value=dict(pid=100,argv=['python','unrelated_training.py'])):
            with self.assertRaisesRegex(RuntimeError,'PID identity'):
                guard.owned_roots(self.config,self.status,self.root/'transition.json')

    def test_retry_budget_and_progress_reset(self):
        snapshot=dict(key='joyai',completed=10,status=self.status)
        state=dict(retries=dict(joyai=dict(completed=10,attempts=3)))
        with patch.object(guard,'stop_owned') as stop:
            self.assertEqual(guard.repair(snapshot,state,self.config,self.root/'config.json',self.root,0)['action'],
                             'needs_attention')
            stop.assert_not_called()
        snapshot['completed']=11
        with patch.object(guard,'owned_roots',return_value=[]),patch.object(guard,'stop_owned',return_value=[]), \
             patch.object(guard,'occupied_gpus',return_value=[]),patch.object(guard.subprocess,'Popen',return_value=Mock(pid=500)) as launch:
            result=guard.repair(snapshot,state,self.config,self.root/'config.json',self.root,1)
        self.assertEqual(result['attempt'],1);self.assertEqual(result['restarted_pid'],500)
        record = dict(pid=123, checked_at=1, health=snapshot, **result)
        self.assertEqual(record['pid'],123)
        self.assertEqual(record['restarted_pid'],500)
        self.assertEqual(launch.call_args.args[0][-2:],['--config',str(self.root/'config.json')])
        self.assertNotIn('pending_repair',state)

    def test_foreign_gpu_process_defers_restart(self):
        state={};snapshot=dict(key='joyai',completed=10,status=self.status)
        with patch.object(guard,'owned_roots',return_value=[]),patch.object(guard,'stop_owned',return_value=[]), \
             patch.object(guard,'occupied_gpus',return_value=['999, GPU-foreign']),patch.object(guard.subprocess,'Popen') as launch:
            result=guard.repair(snapshot,state,self.config,self.root/'config.json',self.root,0)
        self.assertEqual(result['action'],'waiting_for_gpus');launch.assert_not_called()
        self.assertIn('pending_repair',state)

    def test_diagnostic_failure_policy_does_not_retry_failed_workers(self):
        guard.write(self.root/'joyai/parallel_status.json',dict(status='running',failure_policy='skip',
            workers=[dict(rank=0,exit_code=1)]))
        directory=self.root/'joyai/worker_0';directory.mkdir()
        result=self.inspect()
        self.assertEqual(result['verdict'],'healthy')
        self.assertEqual(result['workers'],[])

    def test_probe_failure_in_previous_task_is_not_hidden_by_later_running_task(self):
        (self.root/'suite').mkdir()
        guard.write(self.root/'queue/transition_status.json',{**self.status,'stage':'new_suite'})
        guard.write(self.root/'suite/suite_status.json',dict(jobs=[
            dict(mode='probe',status='incomplete',output=str(self.root/'failed')),
            dict(mode='probe',status='running',output=str(self.root/'later'))]))
        self.assertEqual(self.inspect()['reason'],'incomplete')

    def test_bounded_probe_final_failures_do_not_restart_or_stall_queue(self):
        self.config['next_argv'].append('--probe-retry-errors-once')
        (self.root/'suite').mkdir()
        guard.write(self.root/'queue/transition_status.json',{**self.status,'stage':'new_suite'})
        directory = self.root/'probe';directory.mkdir()
        (directory/'worker_0').mkdir()
        guard.write(self.root/'suite/suite_status.json',dict(jobs=[
            dict(mode='probe',status='incomplete_final',output=str(directory),exit_codes=[1])]))
        result=self.inspect()
        self.assertEqual(result['verdict'],'healthy')
        self.assertEqual(result['workers'],[])

    def test_complete_with_errors_is_terminal(self):
        (self.root/'suite').mkdir()
        guard.write(self.root/'queue/transition_status.json',{**self.status,'status':'complete'})
        guard.write(self.root/'suite/suite_status.json',dict(status='complete_with_errors'))
        self.assertEqual(self.inspect()['verdict'],'complete')


if __name__=='__main__':
    unittest.main()
