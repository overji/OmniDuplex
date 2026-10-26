from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmark.cli.main import build_parser
from benchmark.data.files import read_jsonl
from benchmark.probe.data import file_hash, sample_probes
from benchmark.probe.output import PROTOCOL, parse_reply
from benchmark.evaluation import probe_judge as pj


def sample(mode='pr'):
    result = dict(mode=mode, task='cup_'+mode, ability='object_perception',video='v.mp4',
        question='Tell me when the cup moves.', question_time='00:00', available_tools=[],
        responses=[dict(response='The cup moved.',window_start='00:10',window_end='00:12',
            evidence=[dict(time='00:11',fact='CURRENT EVIDENCE'),dict(time='00:25',fact='FUTURE EVIDENCE')]),
            dict(response='The cup moved again.',window_start='00:20',window_end='00:22')])
    if mode == 'agent':
        result.update(sample_type='positive',available_tools=['note_write'],
            frontend_context={'system_prompt':'Use agent_call.','history':'FUTURE HISTORY'})
        for reference in result['responses']:
            reference.update(role='assistant',tool_calls=[dict(function=dict(name='agent_call',arguments=json.dumps(
                dict(instruction='Log the cup movement.',observed_context='CURRENT CONTEXT'))))])
        result['responses'] += [dict(role='tool',response='FUTURE TOOL RESULT',window_start='00:23',window_end='00:24'),
            dict(role='assistant',response='FUTURE COMPLETION',window_start='00:24',window_end='00:25')]
    return result


def prediction(row, text, complete=True):
    return dict(probe_id=row['probe_id'],pair_id=row['pair_id'],sample_index=row['sample_index'],
        response_index=row['response_index'],side=row['side'],status='complete',output_protocol=PROTOCOL,
        raw_response=text,task_text_generation_complete=complete,decision=parse_reply(text,row['mode'],complete=complete)['decision'])


def answer(response=2,call=None):
    return json.dumps(dict(response_score=response,call_score=call,response_reason='reference match',
                           call_reason='instruction match',error_types=[]))


class ProbeJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def fixture(self,mode='pr'):
        samples=[sample(mode)];data=self.root/'data.json';data.write_text(json.dumps(samples))
        manifest,_=sample_probes(samples[0],0,file_hash(data),Path('/v.mp4'),
            dict(duration=30,has_audio=True,source=dict(path='/v.mp4',size=1,mtime_ns=1)))
        path=self.root/'manifest.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
        run=self.root/'model';run.mkdir()
        return samples,manifest,run

    def args(self,run,*extra):
        return build_parser().parse_args(['probe-judge',str(run),'--data',str(self.root/'data.json'),
            '--manifest',str(self.root/'manifest.jsonl'),'--output-dir',str(self.root/'judged'),
            '--provider','openai','--model','test','--retries','0',*extra])

    def write_predictions(self,run,rows):
        (run/'probe_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_tp_filter_uses_raw_output_and_exact_reference_index(self):
        samples,manifest,_=self.fixture()
        predictions=[prediction(m,t) for m,t in zip(manifest,['<silent>','<silent>','early reply','actual reply'])]
        predictions[-1]['decision']='NO'  # Persisted derived fields are not ground truth.
        rows,index,counts=pj.prepare_rows(samples,manifest,predictions)
        self.assertEqual(counts,dict(TN=1,FN=1,FP=1,TP=1))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['response_index'],1)
        self.assertEqual(rows[0]['payload']['reference']['response'],'The cup moved again.')
        self.assertNotIn('event_index',rows[0]);self.assertNotIn('onset_seconds',rows[0])

    def test_agent_average_and_missing_or_truncated_call_zero(self):
        samples,manifest,_=self.fixture('agent')
        for text,complete,expected,call in [('Logging it.<|task_start|>Log the cup movement.',True,1.75,1.5),
                ('Logging it.',True,1.,None),('Logging it.<|task_start|>Log.',False,1.,None)]:
            with self.subTest(text=text,complete=complete):
                rows,_,_=pj.prepare_rows(samples,manifest,[prediction(manifest[1],text,complete)])
                result=pj.evaluate(rows[0],{'retries':0},'','',lambda *a:(answer(2,call),{},True))
                self.assertEqual(result['task_score'],expected)
                if call is None:self.assertEqual(result['call_score'],0)

    def test_empty_user_response_and_invalid_call_are_local_zero(self):
        samples,manifest,_=self.fixture('agent')
        rows,_,counts=pj.prepare_rows(samples,manifest,[prediction(manifest[1],'<|task_start|>Log it.')])
        request=Mock(side_effect=AssertionError('No API call expected'))
        result=pj.evaluate(rows[0],{'retries':0},'','',request)
        self.assertEqual(counts['TP'],1);self.assertEqual(result['task_score'],0);request.assert_not_called()

    def test_rtd_scores_response_and_does_not_filter_garbage_tp(self):
        samples,manifest,_=self.fixture('rtd')
        rows,_,_=pj.prepare_rows(samples,manifest,[prediction(manifest[1],'<think> repeated 0.000000',False)])
        request=Mock(return_value=(answer(.5),{},True))
        result=pj.evaluate(rows[0],{'retries':0},'','',request)
        self.assertEqual(result['task_score'],.5);self.assertIsNone(result['call_score']);request.assert_called_once()

    def test_payload_is_read_only_and_excludes_future_information(self):
        samples,manifest,_=self.fixture('agent');before=deepcopy(samples)
        rows,_,_=pj.prepare_rows(samples,manifest,[prediction(manifest[1],'Logging.<|task_start|>Log the cup movement.')])
        serialized=json.dumps(rows[0]['payload'])
        self.assertNotIn('FUTURE',serialized);self.assertIn('CURRENT EVIDENCE',serialized)
        self.assertNotIn('observed_context',serialized)
        self.assertEqual(rows[0]['payload']['reference']['tool_calls'][0]['function']['name'],'task_start')
        self.assertEqual(samples,before)

    def test_protocol_duplicates_and_reference_mismatch_rejected(self):
        samples,manifest,_=self.fixture();pred=prediction(manifest[1],'reply')
        with self.assertRaisesRegex(ValueError,'only reply-task-text'):
            pj.prepare_rows(samples,manifest,[{**pred,'output_protocol':'yes_no_v1'}])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            pj.prepare_rows(samples,manifest,[pred,pred])
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            pj.prepare_rows(samples,manifest,[{**pred,'response_index':1}])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            pj.validate_manifest(samples,manifest,'WRONG DATA HASH')

    def test_video_aliases_are_validated_by_resolved_target(self):
        samples,manifest,_=self.fixture();videos=self.root/'videos';videos.mkdir()
        target=videos/'uuid.mp4';target.touch();(videos/'v.mp4').symlink_to(target)
        for row in manifest:row['video']=str(target)
        pj.validate_manifest(samples,manifest,file_hash(self.root/'data.json'),videos)
        for row in manifest:row['video']=str(videos/'other.mp4')
        with self.assertRaisesRegex(ValueError,'alias'):
            pj.validate_manifest(samples,manifest,file_hash(self.root/'data.json'),videos)

    def test_incremental_cache_and_existing_input_change_rejected(self):
        _,manifest,run=self.fixture();preds=[prediction(m,t) for m,t in zip(manifest,['<silent>','reply','<silent>','another reply'])]
        self.write_predictions(run,preds[:2]);args=self.args(run)
        original=pj.evaluate
        request=Mock(return_value=(answer(1.5),{'total_tokens':7},True))
        with patch.object(pj,'resolve_config',return_value=('SECRET KEY','https://example.test')), \
             patch.object(pj,'evaluate',side_effect=lambda row,c,k,s:original(row,c,k,s,request)):
            pj.execute(args);self.assertEqual(request.call_count,1)
            with (run/'probe_predictions.jsonl').open('a') as f:f.write('{"partial":')
            before=(run/'probe_predictions.jsonl').read_bytes()
            pj.execute(args);self.assertEqual(request.call_count,1)
            self.assertEqual((run/'probe_predictions.jsonl').read_bytes(),before)
            self.write_predictions(run,preds);pj.execute(args);self.assertEqual(request.call_count,2)
            preds[1]['raw_response']='Changed reply';self.write_predictions(run,preds)
            with self.assertRaisesRegex(ValueError,'changed or disappeared'):pj.execute(args)
        usage=json.loads((self.root/'judged/model/api_usage.json').read_text())
        self.assertEqual(usage['requests'],2);self.assertEqual(usage['usage_totals']['total_tokens'],14)
        self.assertNotIn('SECRET KEY',(self.root/'judged/model/judge_config.json').read_text())

    def test_api_failure_is_unscored_and_can_retry(self):
        _,manifest,run=self.fixture();self.write_predictions(run,[prediction(manifest[1],'reply')]);args=self.args(run)
        original=pj.evaluate;request=Mock(side_effect=RuntimeError('SECRET REQUEST ERROR'))
        with patch.object(pj,'resolve_config',return_value=('key','https://example.test')), \
             patch.object(pj,'evaluate',side_effect=lambda row,c,k,s:original(row,c,k,s,request)):
            with self.assertRaises(SystemExit):pj.execute(args)
            metrics=json.loads((self.root/'judged/model/judge_metrics.json').read_text())
            self.assertIsNone(metrics['groups']['overall']['task_mean_tp'])
            self.assertEqual(metrics['groups']['overall']['failed_tp'],1)
            self.assertNotIn('SECRET',(self.root/'judged/model/judgments.jsonl').read_text())
            request.side_effect=None;request.return_value=(answer(),{},True)
            pj.execute(args)
        metrics=json.loads((self.root/'judged/model/judge_metrics.json').read_text())
        self.assertEqual(metrics['groups']['overall']['task_mean_tp'],2)

    def test_dry_run_needs_no_credentials_and_writes_no_results(self):
        _,manifest,run=self.fixture();self.write_predictions(run,[prediction(manifest[1],'reply')])
        args=build_parser().parse_args(['probe-judge',str(run),'--data',str(self.root/'data.json'),
            '--manifest',str(self.root/'manifest.jsonl'),'--output-dir',str(self.root/'unused'),'--dry-run'])
        with patch.object(pj,'resolve_config',side_effect=AssertionError('No credentials expected')) as config:
            pj.execute(args);config.assert_not_called()
        self.assertFalse((self.root/'unused').exists())

    def test_overall_is_average_of_per_probe_task_scores(self):
        rows=[dict(mode='agent',ability='test',status='complete',call_applicable=True,response_score=2,call_score=0,task_score=1),
              dict(mode='pr',ability='test',status='complete',call_applicable=False,response_score=2,call_score=None,task_score=2)]
        metrics=pj.summarize(rows,dict(TP=2),2,4)
        self.assertEqual(metrics['groups']['overall']['task_mean_tp'],1.5)
        rows[1].update(status='failed',response_score=None,task_score=None)
        metrics=pj.summarize(rows,dict(TP=2),2,4)
        self.assertIsNone(metrics['groups']['overall']['task_mean_tp'])
        self.assertEqual(metrics['groups']['overall']['scored_task_mean'],1)


if __name__=='__main__':
    unittest.main()
