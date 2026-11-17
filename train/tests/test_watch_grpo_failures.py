import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train.scripts.watch_grpo_failures import scan, secret


class WatchGRPOFailuresTest(unittest.TestCase):
    def test_failure_after_start_is_queued_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'grpo_test'; run.mkdir(); path=run/'status.json'
            config={'status_glob':str(root/'grpo*'/'status.json'),'dead_grace_seconds':30}
            path.write_text(json.dumps({'stage':'training','child_pid':123}))
            state={}
            with patch('train.scripts.watch_grpo_failures.process_identity',return_value='start'):
                scan(config,state,100)
                self.assertEqual(len(state['active']),1)
                path.write_text(json.dumps({'stage':'failed','error':'out of disk'}))
                scan(config,state,101);scan(config,state,102)
            self.assertEqual(len(state['pending']),1)
            self.assertEqual(state['pending'][0]['reason'],'out of disk')

    def test_preexisting_failure_and_normal_completion_do_not_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'grpo_test';run.mkdir();path=run/'status.json'
            config={'status_glob':str(root/'grpo*'/'status.json'),'dead_grace_seconds':30}
            state={};path.write_text(json.dumps({'stage':'failed'}));scan(config,state,100)
            self.assertEqual(state['pending'],[])
            with patch('train.scripts.watch_grpo_failures.process_identity',return_value='start'):
                path.write_text(json.dumps({'stage':'training','child_pid':123}));scan(config,state,101)
                path.write_text(json.dumps({'stage':'complete'}));scan(config,state,102)
            self.assertEqual(state['pending'],[]);self.assertEqual(state['active'],{})

    def test_dead_process_uses_grace_and_key_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'grpo_test';run.mkdir();path=run/'status.json';env=root/'.env'
            env.write_text('MAIL_KEY="example-private-value"\n');self.assertEqual(secret(env),'example-private-value')
            config={'status_glob':str(root/'grpo*'/'status.json'),'dead_grace_seconds':30}
            state={};path.write_text(json.dumps({'stage':'training','child_pid':123}))
            with patch('train.scripts.watch_grpo_failures.process_identity',side_effect=['start',None,None,None,None,None]):
                scan(config,state,100);scan(config,state,101);scan(config,state,132)
            self.assertEqual(len(state['pending']),1)

if __name__=='__main__':unittest.main()
