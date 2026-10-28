"""Portable judging on a CPU-only checkout; API tests use localhost fake services."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from benchmark.cli.main import build_parser
from benchmark.evaluation import judge_bundle as runner
from benchmark.evaluation.judge import fingerprint
from benchmark.evaluation.judge_package import SCHEMA, content_hash, task_id, validate, inspect


def bundle_fixture():
    bundle = dict(schema=SCHEMA, created_at='snapshot', dataset={'sha256': 'dataset'},
                  prompts={'realtime': dict(text='Judge the response. Return JSON.', sha256=fingerprint('Judge the response. Return JSON.'))},
                  runs=[], tasks=[])
    # Two models share one exact scoring request; a third has a missing Agent call.
    for name, agent, matched in [('model_a',False,True),('model_b',False,True),('agent',True,True),('miss',False,False)]:
        bundle['runs'].append(dict(run_id=name, model=name, mode='realtime', variant='baseline'))
        payload = dict(question='Notify me.', question_time='00:00', mode='agent' if agent else 'pr',
                       reference={'response':'Here'}, candidate_response='Here' if matched else '',
                       candidate_tool_calls=[], judge_call=False)
        row = dict(sample_index=0, response_index=0, mode=payload['mode'], ability='object_perception',
                   call_applicable=agent, event_index=0 if matched else None, payload=payload)
        task = dict(run_id=name, rubric='realtime', qa_sha256='qa', video='v.mp4', row=row,
                    api_required=matched, matched=matched)
        task['task_id']=task_id(task);bundle['tasks'].append(task)
    bundle['bundle_id']=content_hash(bundle)
    return bundle


class PackageTests(unittest.TestCase):
    def test_hashes_counts_duplicates_and_modified_payload(self):
        bundle=bundle_fixture();validate(bundle)
        self.assertEqual(inspect(bundle)['unique_api_requests'],2)
        for change in ('payload','duplicate','prompt'):
            b=deepcopy(bundle)
            if change=='payload':b['tasks'][0]['row']['payload']['candidate_response']='changed'
            if change=='duplicate':b['tasks'].append(deepcopy(b['tasks'][0]))
            if change=='prompt':b['prompts']['realtime']['text']='changed'
            b['bundle_id']=content_hash(b)
            with self.assertRaises(ValueError):validate(b)

    def test_dry_run_outside_repo_blocks_gpu_and_model_imports_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.json';source.write_text(json.dumps(bundle_fixture()))
            script='''
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','numpy','PIL','cv2','decord','google'}:
        raise AssertionError('Heavy dependency imported: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from benchmark.cli.main import main
main(['judge-bundle','--input','input.json','--output','results.json','--dry-run'])
'''
            repo=Path(__file__).resolve().parents[2]
            env={**os.environ,'PYTHONPATH':str(repo)}
            result=subprocess.run([sys.executable,'-c',script],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'results.json').exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()),['input.json'])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.input=self.root/'input.json';self.input.write_text(json.dumps(bundle_fixture()))
        self.output=self.root/'results.json'
        self.state={'fail':False,'truncate':0,'requests':[]}
        state=self.state
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                state['requests'].append(body)
                if state['fail']:
                    self.send_response(503);self.end_headers();return
                finish='length' if state['truncate']>0 else 'stop'
                if state['truncate']>0:state['truncate']-=1
                answer=dict(response_score=1.5,call_score=None,response_reason='Correct core.',call_reason='Not applicable',error_types=[])
                result=dict(id='fake-request',model='deepseek-flash',system_fingerprint='test',
                            usage={'prompt_tokens':5,'completion_tokens':2,'total_tokens':7},
                            choices=[dict(finish_reason=finish,message={'content':json.dumps(answer)})])
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def args(self,*extra):
        return build_parser().parse_args(['judge-bundle','--input',str(self.input),'--output',str(self.output),
             '--provider','openai','--model','deepseek-flash','--base-url',f'http://127.0.0.1:{self.server.server_port}',
             '--concurrency','1',*extra])

    def run_score(self,*extra):
        with patch.dict(os.environ,OPENAI_API_KEY='TEST_KEY_NEVER_EXPORT'),patch('benchmark.evaluation.judge.time.sleep'):
            runner.execute(self.args(*extra))

    def test_dedup_rule_zeros_agent_average_and_single_file_resume(self):
        self.run_score()
        result=json.loads(self.output.read_text())
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(self.state['requests']),2)
        by_run={t['run_id']:t for t in result['tasks']}
        self.assertEqual(by_run['agent']['task_score'],.75)
        self.assertEqual(by_run['agent']['call_score'],0)
        self.assertEqual(by_run['miss']['task_score'],0)
        self.assertEqual(by_run['miss']['scoring_source'],'rule')
        self.assertEqual(result['api_usage']['attempts'],2)
        self.assertEqual(result['api_usage']['usage_totals']['total_tokens'],14)
        self.assertEqual(self.state['requests'][0]['thinking'],{'type':'disabled'})
        self.assertEqual(self.state['requests'][0]['max_tokens'],1024)
        self.assertNotIn('TEST_KEY_NEVER_EXPORT',self.output.read_text())
        self.assertEqual(result['requests'][0]['result']['usage_attempts'][0]['usage']['_response_metadata']['model'],'deepseek-flash')
        Path(str(self.output)+'.progress.jsonl').unlink()
        self.run_score();self.assertEqual(len(self.state['requests']),2)

    def test_failures_stay_null_and_require_explicit_retry_failed(self):
        self.state['fail']=True
        with self.assertRaises(SystemExit) as stop:self.run_score()
        self.assertEqual(stop.exception.code,2)
        first=json.loads(self.output.read_text());self.assertEqual(first['status'],'incomplete')
        self.assertEqual(first['summary']['model_a']['overall']['response_mean'],None)
        self.assertIsNone(next(t for t in first['tasks'] if t['run_id']=='agent')['call_score'])
        self.assertEqual(len(self.state['requests']),6)
        self.state['fail']=False
        with self.assertRaises(SystemExit):self.run_score()
        self.assertEqual(len(self.state['requests']),6)
        self.run_score('--retry-failed')
        self.assertEqual(len(self.state['requests']),8)
        result=json.loads(self.output.read_text());self.assertEqual(result['status'],'complete')
        self.assertEqual(result['api_usage']['attempts'],8)
        self.assertEqual(result['api_usage']['failed_attempts'],6)

    def test_truncation_is_retried_and_interrupted_journal_tail_is_recovered(self):
        self.state['truncate']=1
        self.run_score()
        self.assertEqual(len(self.state['requests']),3)
        journal=Path(str(self.output)+'.progress.jsonl')
        with journal.open('ab') as stream:stream.write(b'{"interrupted":')
        self.run_score();self.assertEqual(len(self.state['requests']),3)
        self.assertTrue(journal.read_bytes().endswith(b'\n'))

    def test_changed_judge_configuration_and_tampered_return_are_rejected(self):
        self.run_score()
        with self.assertRaisesRegex(ValueError,'different input or judge configuration'):
            self.run_score('--max-tokens','2048')
        self.assertEqual(len(self.state['requests']),2)
        journal=Path(str(self.output)+'.progress.jsonl');journal.unlink()
        result=json.loads(self.output.read_text());result['requests'][0]['result']['response_score']=3
        self.output.write_text(json.dumps(result))
        with self.assertRaises(ValueError):self.run_score()


if __name__=='__main__':unittest.main()
