"""External protocol contracts without network, weights, GPU or real Judge calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from benchmark.cli.main import build_parser
from benchmark.data.files import write_json
from benchmark.data.interactscope import convert, build_prompt, window_conflicts, prepare, SOURCE
from benchmark.external.reporting import aggregate, report
from benchmark.external.scoring import predicted_tools, tool_judgment, judge
from benchmark.evaluation.judge import prepare_rows
from benchmark.runtime.runner import SampleContext


def row(task='T2', sub='A', identifier='one', video='videos/v.mp4'):
    return dict(interaction_id=identifier, task=task, sub_task_type=sub, task_type='proactive_notifying',
                query_type='text',query_text='Notify me when the cup moves.',query_time_ms=2400.,
                query_image='',query_audio='',qa_language='en',target_group='visually_impaired',
                video=video,video_duration_sec=6.3,video_fps=30.,source_start_ms=10000.,source_end_ms=16300.,
                evaluation_splits=['table3' if task=='T8' else 'table2'],
                source_attribution=dict(source='example',original_video_url='https://example.com/id',platform_video_id='id'),
                responses=[dict(notify_trigger_time_ms=4000.,valid_response_window_ms=None,
                                assistant_reply_text='The cup moved.',trigger_reason='SECRET EVENT',expected_behavior='respond')])


def prediction(sample, events, index=0):
    return dict(sample_index=index,status='complete',observed_seconds=sample['duration_seconds'],
                events=events,system_prompt_sha256='same',**{k:sample[k] for k in ['video','question','task']})


class InteractScopeConversion(unittest.TestCase):
    def test_milliseconds_once_source_crop_does_not_shift_clip_times(self):
        source=row();sample=convert(source)
        self.assertEqual(sample['question_time'],2.4)
        self.assertEqual([sample['responses'][0][k] for k in ['window_start','window_end']],[2.,6.])
        self.assertEqual(sample['source_sample'],source)
        self.assertEqual(sample['id'],'one')

    def test_t8_windows_and_query_free_context(self):
        a=row('T8','A');a['query_text']='';a['query_time_ms']=None
        a['responses'][0]['valid_response_window_ms']=[100.,1300.]
        sample=convert(a)
        self.assertTrue(sample['query_free'])
        self.assertEqual(sample['question_time'],0)
        self.assertEqual(sample['question'],'')
        self.assertEqual(sample['responses'][0]['window_end'],1.3)
        for sub in ['B','C']:
            sample=convert(row('T8',sub))
            self.assertEqual(sample['responses'][0]['window_start'],2.4)
        self.assertIn('cannot see',sample['frontend_context']['system_prompt'])

    def test_invalid_times_not_guessed(self):
        for value in (None,True,float('nan')):
            source=row();source['query_time_ms']=value
            with self.assertRaises(ValueError):convert(source)
        source=row('T8','A')
        with self.assertRaises(ValueError):convert(source)

    def test_tool_names_are_not_instruction_targets(self):
        source=row('T1');source['responses'][0]['assistant_reply_text']='<call:image_search>Reference search result'
        sample=convert(source)
        self.assertEqual(sample['mode'],'tool')
        self.assertEqual(sample['responses'][0]['reference_tool_names'],['image_search'])
        self.assertNotIn('tool_calls',sample['responses'][0])
        with self.assertRaises(ValueError):convert(row('T1'))

    def test_prompt_whitelist_no_future_reference_or_tool_distribution(self):
        sample=convert(row())
        sample['source_sample']['query_text']='SOURCE SECRET'
        sample['responses'][0]['response']='REFERENCE SECRET'
        prompt=build_prompt(sample)
        for secret in ['SOURCE SECRET','REFERENCE SECRET','SECRET EVENT','4000']:
            self.assertNotIn(secret,prompt)
        with TemporaryDirectory() as directory:
            context=SampleContext(0,sample,'OLD BASE',Path(directory),agent_output_protocol='task-text')
            self.assertEqual(context.build_prompt(),prompt)

    def test_overlaps_kept_and_zero_length_endpoint_touch_detected(self):
        sample=convert(row())
        sample['responses'].append(dict(sample['responses'][0],window_start=6,window_end=8))
        self.assertEqual(window_conflicts(sample)[0]['overlap_seconds'],0)
        self.assertEqual(len(sample['responses']),2)
        sample['responses'][1]['window_start']=5
        self.assertEqual(window_conflicts(sample)[0]['overlap_seconds'],1)

    def test_prepare_reports_missing_media_preserves_original_order_and_hashes(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset';dataset.mkdir();(dataset/'videos').mkdir()
            video=dataset/'videos/v.mp4';video.write_bytes(b'video fixture')
            source=[row(),row(identifier='missing',video='videos/missing.mp4')]
            write_json(dataset/'annotations.json',dict(schema_version='omni-interactscope-1.1',interactions=source))
            specs=[dict(path='videos/v.mp4',bytes=video.stat().st_size,sha256=hashlib.sha256(video.read_bytes()).hexdigest())]
            (dataset/'file_manifest.jsonl').write_text(json.dumps(specs[0])+'\n')
            code=root/'code/configs';code.mkdir(parents=True);(code/'tools.yaml').write_text('tools:\n  image_search:\n    desc: Search images.\n')
            cfg=dict(dataset_root=str(dataset),media_workers=1,official_code=str(root/'code'),dataset_revision='pin',official_code_revision='codepin')
            with patch('benchmark.data.interactscope.probe_video',return_value=dict(duration=6.3,has_audio=True)):
                meta=prepare(cfg,root/'output')
            self.assertEqual(meta['runnable_qa'],1)
            self.assertEqual(meta['missing_videos'],1)
            self.assertFalse(meta['full_coverage'])
            mapped=json.loads((root/'output/source_mapping.json').read_text())
            self.assertEqual([r['sample_index'] for r in mapped],[0,None])
            self.assertIn('missing_media',mapped[1]['reasons'])
            self.assertEqual(json.loads((root/'output/runnable_qa.json').read_text())[0]['source_sample'],source[0])
            before=(root/'output/runnable_qa.json').read_bytes()
            source[0]['query_text']='Changed request'
            write_json(dataset/'annotations.json',dict(schema_version='omni-interactscope-1.1',interactions=source))
            with patch('benchmark.data.interactscope.probe_video',return_value=dict(duration=6.3,has_audio=True)):
                with self.assertRaisesRegex(ValueError,'provenance/coverage changed'):
                    prepare(cfg,root/'output')
            self.assertEqual((root/'output/runnable_qa.json').read_bytes(),before)
            specs[0].update(bytes=1,sha256='not-the-original-video')
            (dataset/'file_manifest.jsonl').write_text(json.dumps(specs[0])+'\n')
            from benchmark.data.interactscope import digest
            def no_video_hash(path):
                self.assertNotEqual(Path(path).suffix,'.mp4')
                return digest(path)
            with patch('benchmark.data.interactscope.probe_video',return_value=dict(duration=6.3,has_audio=True)),patch('benchmark.data.interactscope.digest',side_effect=no_video_hash):
                decoded=prepare(dict(cfg,media_verification='decode'),root/'decoded')
            self.assertEqual(decoded['runnable_qa'],1)
            self.assertEqual(decoded['media_verification'],'decode')


class InteractScopeScores(unittest.TestCase):
    def test_only_explicit_markers_or_native_calls_count_as_tools(self):
        self.assertEqual(predicted_tools(dict(text='I will use image_search.')),[])
        self.assertEqual(predicted_tools(dict(text='<call:image_search><call:unknown>')),['image_search','unknown'])
        self.assertEqual(predicted_tools(dict(text='',tool_calls=[dict(name='image_search')])),['image_search'])
        self.assertEqual(predicted_tools(dict(text='["image_search", "wrong"]')),['image_search','wrong'])
        self.assertEqual(predicted_tools(dict(text='<tool_call>{"name":"image_search"}</tool_call>')),['image_search'])

    def test_exact_tool_set_and_missing_or_extra_names(self):
        source=row('T1');source['responses'][0]['assistant_reply_text']='<call:image_search>'
        sample=convert(source)
        for text,score in [('<call:image_search>',2),('<call:image_search><call:web_search>',0),('image_search',0),('',0)]:
            pred=prediction(sample,[dict(onset_seconds=4,text=text)])
            rows,_=prepare_rows([sample],[pred])
            result=tool_judgment(rows[0],sample,pred)
            self.assertEqual(result['response_score'],score)
            self.assertFalse(result['call_applicable'])
        pred=prediction(sample,[dict(onset_seconds=1,text='<call:image_search>')])
        rows,_=prepare_rows([sample],[pred])
        self.assertEqual(tool_judgment(rows[0],sample,pred)['response_score'],0)

    def test_extra_responses_reduce_both_f1_and_scored_f1(self):
        sample=convert(row())
        event=dict(onset_seconds=4,text='The cup moved.')
        pred=prediction(sample,[event,dict(event,onset_seconds=6.2)])
        judgment=dict(sample_index=0,response_index=0,status='complete',response_score=2,call_score=None,call_applicable=False)
        groups=aggregate([sample],[pred],[judgment])
        self.assertAlmostEqual(groups['text']['f1'],2/3)
        self.assertAlmostEqual(groups['text']['scored_f1'],2/3)
        self.assertNotIn('tool',groups)
        self.assertIsNone(aggregate([sample],[pred],[])['text']['scored_f1'])

    def test_one_output_cannot_hit_two_overlapping_targets(self):
        sample=convert(row());sample['responses'].append(deepcopy(sample['responses'][0]))
        result=aggregate([sample],[prediction(sample,[dict(onset_seconds=4,text='Here')])],[])['text']
        self.assertEqual(result['hits'],1)
        self.assertEqual(result['windows'],2)

    def test_live_report_uses_all_configured_models_common_cohort(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);sample=convert(row());write_json(root/'data.json',[sample])
            write_json(root/'manifest.json',dict(source_qa=1479,full_coverage=False))
            config=dict(output=str(root),data=str(root/'data.json'),protocol='test',source_sha256='source',data_sha256='data',
                        dataset_revision='pin',models=[dict(label='base'),dict(label='sft')],judge=dict(model='Qwen27B'))
            write_json(root/'config.json',config);run=root/'runs/base';run.mkdir(parents=True)
            (run/'predictions.jsonl').write_text(json.dumps(prediction(sample,[dict(onset_seconds=4,text='Here')]))+'\n')
            result=report(root/'config.json')
            self.assertEqual(result['paired_qa'],0)
            self.assertEqual(result['individual']['base']['text']['hits'],1)

    def test_tool_judging_never_calls_llm(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=row('T1');source['responses'][0]['assistant_reply_text']='<call:image_search>'
            sample=convert(source);write_json(root/'data.json',[sample]);(root/'prompt.txt').write_text('Judge')
            run=root/'runs/base';run.mkdir(parents=True)
            (run/'predictions.jsonl').write_text(json.dumps(prediction(sample,[dict(onset_seconds=4,text='<call:image_search>')]))+'\n')
            cfg=dict(output=str(root),data=str(root/'data.json'),data_sha256='digest',protocol='test',
                     judge=dict(model='Qwen27B',system_prompt=str(root/'prompt.txt'),max_tokens=512,timeout=1,retries=0,concurrency=1))
            with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('No API for tools')):
                result=judge(cfg,'base','http://local/v1')
            self.assertEqual(result,dict(windows=1,failed=0))

    def test_judge_replicas_and_current_reference_only(self):
        import io
        with TemporaryDirectory() as directory:
            root=Path(directory);source=row()
            source['responses'].append(dict(source['responses'][0],notify_trigger_time_ms=5000.,assistant_reply_text='FUTURE SECRET'))
            sample=convert(source);write_json(root/'data.json',[sample]);(root/'prompt.txt').write_text('Judge')
            run=root/'runs/base';run.mkdir(parents=True)
            pred=prediction(sample,[dict(onset_seconds=t,text='Here') for t in (4,5)])
            (run/'predictions.jsonl').write_text(json.dumps(pred)+'\n')
            cfg=dict(output=str(root),data=str(root/'data.json'),data_sha256='digest',protocol='test',
                     judge=dict(model='Qwen27B',system_prompt=str(root/'prompt.txt'),max_tokens=512,timeout=1,retries=0,concurrency=1))
            calls=[]
            def opened(req,timeout):
                calls.append((req.full_url,json.loads(req.data)))
                answer=json.dumps(dict(response_score=2,call_score=None,response_reason='ok',call_reason='N/A',error_types=[]))
                return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content=answer),finish_reason='stop')])).encode())
            with patch('urllib.request.OpenerDirector.open',side_effect=opened):
                result=judge(cfg,'base',['http://one/v1','http://two/v1'])
            self.assertEqual(result['failed'],0)
            self.assertEqual([url for url,_ in calls],['http://one/v1/chat/completions','http://two/v1/chat/completions'])
            self.assertNotIn('FUTURE SECRET',calls[0][1]['messages'][1]['content'])


class InteractScopeRuntime(unittest.TestCase):
    def test_export_marker_cannot_point_to_another_checkpoint(self):
        from benchmark.external.runtime import export_ready
        with TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.mkdir()
            write_json(model/'full_training_model.json',dict(source_checkpoint=str(root/'wrong/checkpoint-final')))
            marker=root/'ready.json'
            write_json(marker,dict(model=str(model),metadata_sha256=hashlib.sha256((model/'full_training_model.json').read_bytes()).hexdigest()))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                export_ready(dict(model_path=str(model),export_ready_marker=str(marker),training_output=str(root/'selected')))

    def test_cli_requires_explicit_model_and_dry_run_does_not_start_jobs(self):
        args=build_parser().parse_args(['external','run','--config','/config','--model','sft','--dry-run'])
        self.assertIsNone(args.gpus)
        from benchmark.external.runtime import dry_run
        cfg=dict(output='/tmp/out',python='/python',dataset_root='/data',data='/qa',system_prompt='/prompt',inference={})
        job=dict(label='sft',model_path='/external/model',training_output='/external/train')
        with patch('benchmark.external.runtime.validate'),patch('benchmark.external.runtime.training_ready',side_effect=FileNotFoundError),patch('subprocess.Popen',side_effect=AssertionError):
            result=dry_run(cfg,[job],None)
        self.assertFalse(result[0]['ready'])
        self.assertEqual(result[0]['state'],'selected_training_final_not_ready')

    def test_request_inserted_at_fractional_time_before_following_media(self):
        self.stream(query_free=False)

    def test_query_free_sends_no_synthetic_user_question(self):
        self.stream(query_free=True)

    def stream(self,query_free):
        import numpy as np
        from benchmark.adapters import minicpmo
        with TemporaryDirectory() as directory:
            root=Path(directory);source=row('T8','A') if query_free else row()
            if query_free:
                source.update(query_time_ms=None,query_text='')
                source['responses'][0]['valid_response_window_ms']=[100.,1300.]
            sample=convert(source);trace=[]
            def prefill(**kwargs):
                trace.append('question' if 'text_list' in kwargs else 'media')
                return dict(success=True)
            def generate(**kwargs):
                trace.append('generate');return dict(text='Here',is_listen=False,end_of_turn=True)
            model=NS(prepare=Mock(),streaming_prefill=prefill,streaming_generate=generate)
            args=NS(seed=42,videos_dir=root,cache_dir=root,max_seconds=0,decode_mode='sampling',max_speak_tokens=20)
            context=SampleContext(0,sample,'Base',root)
            meta=dict(duration=6.3,frames=[],frame_times=[])
            with patch.object(minicpmo,'media_cache',return_value=(root,meta,np.zeros(100800))):
                result=minicpmo.run_sample(args,NS(model=model),context)
            if query_free:
                self.assertNotIn('question',trace)
                self.assertEqual(result.events[0]['onset_seconds'],1.)
            else:
                q=trace.index('question')
                self.assertEqual(trace[q:q+3],['question','media','generate'])
                self.assertEqual(result.events[0]['onset_seconds'],3.)
                raw=[json.loads(line) for line in (root/'sample_000.chunks.jsonl').read_text().splitlines()]
                self.assertIn(2.4,[r['video_time_seconds'] for r in raw])


if __name__=='__main__':unittest.main()
