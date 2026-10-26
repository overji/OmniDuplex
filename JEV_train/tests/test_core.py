from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from JEV_train.src.config import load,ROOT,PROTOCOL
from JEV_train.src.data import record,sampled_labels,prepare,load_rows
from JEV_train.src.backend import parse_reply
from JEV_train.src.metrics import binary_metrics,evaluate
from JEV_train.src.media import MediaUnit
from JEV_train.src.pipeline import run_session


def sample(video='v.mp4'):
    return dict(video=video,mode='pr',task='test_pr',ability='test',question='Tell me when red appears',question_time='00:00',
        responses=[dict(response='Red appeared.',window_start='00:03',window_end='00:05',trigger_time='00:04'),
                   dict(response='Red again.',window_start='00:08',window_end='00:09',trigger_time='00:08')])


class DataTests(unittest.TestCase):
    def test_multiple_events_and_masking_are_causal(self):
        s=sample();s['question_time']='00:01';before=deepcopy(s)
        row,reason=record(s,0,10)
        self.assertIsNone(reason);self.assertEqual(s,before)
        self.assertEqual(row['labels'],[-100,0,-100,1,-100,0,0,1,-100,0])
        self.assertEqual([e['second'] for e in row['events']],[4,8])
        self.assertEqual(record(s,0,8)[1],'event_exceeds_horizon')
        s['responses'][0]['evidence']=[dict(time='00:06')]
        self.assertEqual(record(s,0,10)[1],'noncausal_event')
    def test_fractional_question_and_collision(self):
        s=sample();s['question_time']='00:02.500';s['responses'][0].update(window_start='00:03',trigger_time='00:03')
        row,_=record(s,0,10);self.assertEqual(row['events'][0]['second'],4)
        s['responses'][1].update(window_start='00:03',trigger_time='00:04')
        self.assertEqual(record(s,0,10)[1],'colliding_events')
    def test_negative_sampling_changes_only_supervision(self):
        row,_=record(sample(),4,30);config=load(ROOT/'JEV_train/configs/default.json')
        before=deepcopy(row);a=sampled_labels(row,config)
        self.assertEqual(a,sampled_labels(row,config));self.assertEqual(row,before)
        self.assertEqual(a.count(1),2);self.assertEqual(a.count(0),8)
        self.assertTrue(all(x==-100 or x==y for x,y in zip(a,row['labels'])))
    def test_agent_conversion_and_optional_payload(self):
        s=sample();s.update(mode='agent',available_tools=['product_search'])
        s['responses']=s['responses'][:1];s['responses'][0].update(role='assistant',tool_calls=[dict(function=dict(name='agent_call',
            arguments=json.dumps({'instruction':'search caps','observed_context':'private evidence'})))])
        row,_=record(s,0,10);text=row['events'][0]['target']
        self.assertEqual(text,'Red appeared.<|task_start|>search caps')
        self.assertNotIn('observed_context',text)
        self.assertEqual(parse_reply(text)['tool_calls'][0]['function']['arguments'],{'instruction':'search caps'})
        self.assertEqual(parse_reply('Hello')['tool_calls'],[])
        self.assertEqual(parse_reply(text,complete=False)['tool_calls'],[])
        self.assertEqual(parse_reply('Hello<|agent|>{}')['agent_parse_status'],'malformed')
        for invalid in ('Hello<|task_start|>', '<|task_start|>search caps',
                        'Hello<|task_start|>search<|task_start|>caps'):
            self.assertEqual(parse_reply(invalid)['tool_calls'],[])
    def test_prepare_video_disjoint_and_original_file_unchanged(self):
        config=load(ROOT/'JEV_train/configs/default.json')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config['videos_dir']=str(root);source=root/'data.json'
            records=[]
            for i in range(20):
                (root/f'{i}.mp4').touch();records.extend([sample(f'{i}.mp4'),sample(f'{i}.mp4')])
            source.write_text(json.dumps(records));original=source.read_bytes()
            with patch('benchmark.data.media.media_cache',return_value=(root,{'duration':10},None)):
                prepare(source,root/'prepared',config)
            self.assertEqual(source.read_bytes(),original)
            manifest=json.loads((root/'prepared/manifest.json').read_text());splits=[set(v) for v in manifest['videos'].values()]
            self.assertFalse(any(a&b for i,a in enumerate(splits) for b in splits[i+1:]))
            self.assertTrue(load_rows(root/'prepared','train',config))
            manifest['protocol']='old';(root/'prepared/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_rows(root/'prepared','train',config)


class PipelineTests(unittest.TestCase):
    def test_evaluation_preserves_failed_partial_outputs_and_emits_report(self):
        from JEV_train.src.pipeline import run
        row,_=record(sample(),0,10)
        class Gate:
            def __init__(self,*args):pass
            def start(self,s):pass
            def decide(self,u):return u.second==4,float(u.second==4)
            def feedback(self,text):pass
        class Backend:
            checked=True
            def start_session(self,s):pass
            def append_media(self,u):
                if u.second==6:raise ValueError('context overflow')
            def generate_response(self):return parse_reply('A reply')
            def close_session(self):pass
        config=load(ROOT/'JEV_train/configs/default.json');config['classifier_adapter']='mock'
        with tempfile.TemporaryDirectory() as tmp, \
             patch('JEV_train.src.gate.load',return_value=object()), \
             patch('JEV_train.src.gate.GateSession',Gate), \
             patch('JEV_train.src.media.units',return_value=iter(MediaUnit(t) for t in range(1,11))), \
             patch.dict('JEV_train.src.backend.BACKENDS',{'qwen25_omni':lambda c:Backend()}):
            output=Path(tmp)/'evaluation';run([row],config,output)
            result=json.loads((output/'predictions.json').read_text())[0]
            self.assertEqual(result['status'],'failed');self.assertEqual(len(result['trace']),5)
            self.assertEqual(len(result['events']),1)
            self.assertEqual(json.loads((output/'metrics.json').read_text())['completed_qa'],0)
            self.assertEqual(json.loads((output/'run_config.json').read_text())['agent_call_schema'],'task-start')
            self.assertTrue((output/'report.md').is_file());self.assertTrue((output/'predictions.jsonl').is_file())
    def test_media_once_before_reply_and_queue_preserves_seconds(self):
        row,_=record(sample(),0,10);calls=[]
        class Gate:
            def start(self,s):self.n=0;self.assert_no_refs='responses' not in s
            def decide(self,u):self.n=u.second;calls.append(('gate',u.second));return u.second in (4,8),float(u.second in (4,8))
            def feedback(self,text):calls.append(('feedback',self.n))
        class Backend:
            checked=True
            def start_session(self,s):self.n=0
            def append_media(self,u):self.n+=1;assert self.n==u.second;calls.append(('media',u.second))
            def generate_response(self):calls.append(('generate',self.n));return parse_reply('A reply')
            def close_session(self):calls.append(('close',self.n))
        gate=Gate();result=run_session(row,(MediaUnit(t) for t in range(1,11)),gate,Backend())
        self.assertTrue(gate.assert_no_refs);self.assertEqual(len(result['probabilities']),10)
        self.assertEqual([x for x in calls if x[0]=='media'],[('media',t) for t in range(1,11)])
        for t in (4,8):self.assertLess(calls.index(('media',t)),calls.index(('generate',t)))
        self.assertEqual(evaluate([row],[result],.5)['timing']['groups']['overall']['hits'],2)
    def test_producer_failure_is_not_lost(self):
        class Gate:
            def start(self,s):pass
        class Backend:
            def start_session(self,s):pass
            def close_session(self):pass
        def source():raise ValueError('broken media');yield
        with self.assertRaisesRegex(ValueError,'broken media'):run_session({'sample':sample()},source(),Gate(),Backend())


class MetricTests(unittest.TestCase):
    def test_auc_ties_masks_and_degenerate_labels(self):
        m=binary_metrics([0,1,0,1,-100],[.1,.8,.2,.9,1],.5)
        self.assertEqual(m['f1'],1);self.assertEqual(m['auroc'],1);self.assertEqual(m['auprc'],1)
        self.assertEqual(binary_metrics([0,1],[.5,.5],.5)['auroc'],.5)
        self.assertIsNone(binary_metrics([1],[.9],.5)['auroc'])
    def test_rows_receive_gradient_and_old_rows_do_not(self):
        import torch
        from JEV_train.src.rows import DecisionEmbedding
        base=torch.nn.Embedding(10,8);base.requires_grad_(False);before=base.weight.detach().clone()
        embed=DecisionEmbedding(base,[8,9]);head=torch.nn.Linear(8,2)
        loss=torch.nn.functional.cross_entropy(head(embed(torch.tensor([8,9]))),torch.tensor([0,1]));loss.backward()
        self.assertIsNone(base.weight.grad);self.assertGreater(embed.rows.grad.abs().sum(),0)
        self.assertGreater(head.weight.grad.abs().sum(),0);torch.testing.assert_close(base.weight,before)


if __name__=='__main__':unittest.main()
