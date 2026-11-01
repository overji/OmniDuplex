"""OmniPro source semantics, tolerance boundaries, prompts and final readiness."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from benchmark.data.omnipro import convert, prepare, TASKS
from benchmark.evaluation.timing import match_windows
from benchmark.data.prompts import build_system_prompt,user_request
from benchmark.runtime.omnipro_queue import final_ready


def source(task='instant_event_alert', clock=10):
    return dict(id='one',task=task,video_id='v',video_path='raw_videos/v.mp4',duration=30,
        question='Tell me when it happens.',question_time='00:00',question_time_sec=0,
        ground_truth=[dict(trigger_time='00:10',trigger_time_sec=clock,response='SECRET REF',
                           trigger_type='speech',count=3,event_description='SECRET EVENT')])


class OmniProTests(unittest.TestCase):
    def test_lossless_conversion_no_prompt_leakage_and_all_tasks(self):
        for task in TASKS:
            row=source(task);before=copy.deepcopy(row);sample=convert([row])[0]
            self.assertEqual(row,before)
            self.assertEqual(sample['source_sample'],row)
            self.assertEqual(sample['responses'][0]['details'],row['ground_truth'][0])
            self.assertEqual((sample['mode'],sample['cognitive_level']),TASKS[task])
            prompt=build_system_prompt('base',sample)+user_request(sample)
            self.assertNotIn('SECRET',prompt)
            self.assertNotIn('trigger_time',prompt)

    def test_inclusive_three_second_tolerance_and_duplicate_onsets(self):
        sample=convert([source()])[0]
        for onset,hits in [(6.999,0),(7,1),(10,1),(13,1),(13.001,0)]:
            self.assertEqual(match_windows(sample,[dict(onset_seconds=onset)])['hits'],hits)
        s=match_windows(sample,[dict(onset_seconds=8),dict(onset_seconds=9)])
        self.assertEqual((s['hits'],s['extra_predictions']),(1,1))
        sample['responses'].append(dict(response='next',window_start=9,window_end=15))
        s=match_windows(sample,[dict(onset_seconds=10)])
        self.assertEqual((s['hits'],s['misses']),(1,1))
        row=source(clock=1);r=convert([row])[0]['responses'][0]
        self.assertEqual((r['window_start'],r['window_end']),(0,4))

    def test_audit_numeric_timestamp_anomaly_and_source_overlap(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);(p/'raw_videos').mkdir();(p/'raw_videos/v.mp4').touch()
            row=source(clock=50);row['ground_truth'][0]['trigger_time']='broken'
            src=p/'source.json';src.write_text(json.dumps([row]))
            meta=p/'metadata.json';meta.write_text(json.dumps(dict(videos=[dict(original_file='v.mp4')])))
            with patch('benchmark.data.omnipro.video_duration',return_value=30):
                m=prepare(src,p,p/'output',meta)
                self.assertEqual(m['references'],1)
                self.assertTrue(m['anomalies'][0]['window_unreachable'])
                self.assertEqual(m['timestamp_annotations'][0]['used_seconds'],50)
                self.assertEqual(m['training_overlap']['shared_source_video_ids'],['v'])
                self.assertEqual(prepare(src,p,p/'output',meta),m)
                row['question']='Changed';src.write_text(json.dumps([row]))
                with self.assertRaisesRegex(ValueError,'changed'):prepare(src,p,p/'output',meta)

    def test_missing_video_is_not_silent_exclusion(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);src=p/'source.json';src.write_text(json.dumps([source()]))
            with self.assertRaisesRegex(ValueError,'Missing'):prepare(src,p,p/'out')

    def test_final_checkpoint_needs_all_rank_reports_and_dead_training(self):
        with TemporaryDirectory() as directory:
            p=Path(directory);job=dict(training_output=str(p),training_pid=123,training_world_size=2,expected_steps=500)
            self.assertFalse(final_ready(job,lambda pid:None))
            c=p/'checkpoint-final';(c/'state').mkdir(parents=True);(c/'training_config.json').write_text('{}')
            for rank in range(2):
                (c/f'rng_rank{rank}.pt').touch();(c/'state'/f'{rank}_optim_states.pt').touch()
                (p/f'memory_report.rank{rank}.json').write_text(json.dumps(dict(checkpoint_saved=True,optimizer_steps=500,last_loss=2)))
            self.assertTrue(final_ready(job,lambda pid:None))
            self.assertFalse(final_ready(job,lambda pid:dict(pid=pid)))

    def test_cli(self):
        from benchmark.cli.main import build_parser
        args=build_parser().parse_args(['omnipro','prepare','--source','a','--videos-dir','b','--output-dir','c'])
        self.assertEqual(args.omnipro_command,'prepare')

class OmniProReportingTests(unittest.TestCase):
    def test_report_uses_same_complete_qa_and_excludes_prompt_mismatch(self):
        from benchmark.evaluation.omnipro import report
        with TemporaryDirectory() as directory:
            p=Path(directory);samples=convert([source()]);samples[0]['duration_seconds']=30
            data=p/'data.json';data.write_text(json.dumps(samples))
            for label, prompt in [('base','a'),('sft','b')]:
                run=p/'runs'/label;run.mkdir(parents=True)
                row=dict(sample_index=0,status='complete',events=[dict(onset_seconds=10,text='answer')],
                         observed_seconds=30,system_prompt_sha256=prompt)
                (run/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            config=p/'queue.json';config.write_text(json.dumps(dict(output=str(p),data=str(data),models=[dict(label='base'),dict(label='sft')])))
            report(config)
            result=json.loads((p/'comparison.json').read_text())
            self.assertEqual(result['paired_qa'],0)
            self.assertEqual(result['excluded_mismatches'],[0])
            r=json.loads((p/'runs/sft/predictions.jsonl').read_text());r['system_prompt_sha256']='a'
            (p/'runs/sft/predictions.jsonl').write_text(json.dumps(r)+'\n')
            report(config);result=json.loads((p/'comparison.json').read_text())
            self.assertEqual(result['paired_qa'],1)
            self.assertEqual(result['paired']['base']['groups']['overall']['timing_f1'],1)

    def test_local_judge_balances_four_endpoints_and_caches_results(self):
        from benchmark.evaluation.omnipro import score_run
        from unittest.mock import MagicMock
        with TemporaryDirectory() as directory:
            p=Path(directory);samples=convert([source()]);sample=samples[0]
            sample['responses'] *= 4
            data=p/'data.json';data.write_text(json.dumps(samples));run=p/'runs/base';run.mkdir(parents=True)
            prediction=dict(sample_index=0,status='complete',task=sample['task'],video=sample['video'],question=sample['question'],
                            events=[dict(onset_seconds=10,text='answer') for _ in range(4)])
            (run/'predictions.jsonl').write_text(json.dumps(prediction)+'\n');prompt=p/'prompt.txt';prompt.write_text('judge')
            urls=[f'http://127.0.0.1:{i}/v1' for i in (8914,8915,8916,8917)]
            cfg=dict(output=str(p),data=str(data),judge=dict(system_prompt=str(prompt),model='Qwen3.8-27B',base_url=urls[0],base_urls=urls,
                max_tokens=512,timeout=2,retries=0,concurrency=4))
            called=[]
            def request(req,timeout):
                called.append(req.full_url)
                answer=dict(response_score=1.5,call_score=None,response_reason='ok',call_reason='NA',error_types=[])
                response=MagicMock();response.__enter__.return_value=__import__('io').BytesIO(json.dumps(dict(choices=[dict(message=dict(content=json.dumps(answer)),finish_reason='stop')],usage={})).encode())
                return response
            opener=MagicMock();opener.open.side_effect=request
            with patch('train.src.infrastructure.judge.identity',return_value={'model':'test'}),patch('benchmark.evaluation.omnipro.urllib.request.build_opener',return_value=opener):
                self.assertEqual(score_run(cfg,'base'),'complete')
                self.assertEqual(set(called),{u+'/chat/completions' for u in urls})
                self.assertEqual(score_run(cfg,'base'),'complete')
                self.assertEqual(len(called),4)

    def test_pause_snapshots_all_owned_roots_before_signalling(self):
        from benchmark.runtime.omnipro_queue import Queue
        from unittest.mock import MagicMock
        with TemporaryDirectory() as directory:
            p=Path(directory);(p/'watchdog').mkdir()
            previous=p/'queue_transition.json';previous.write_text(json.dumps(dict(output=str(p))))
            config_path=p/'insertion.json';config_path.touch()
            queue=Queue.__new__(Queue);queue.path=config_path;queue.config=dict(previous_transition=str(previous),stop_grace_seconds=2,gpus=['4'])
            queue.paused=False;queue.watch=MagicMock();queue.wait_free=MagicMock();queue.state=MagicMock()
            queue.watch.read.side_effect=[dict(pid=1),dict(pid=2,child_pid=3)]
            entries={1:dict(pid=1,argv=['watch_benchmark_queue.py',str(previous)]),
                     2:dict(pid=2,argv=['transition_task_text_queue.py',str(previous)]),
                     3:dict(pid=3,argv=['run_prepared_suite.py',str(p/'commands.json')])}
            queue.watch.process.side_effect=lambda pid:entries[pid]
            queue.pause_previous()
            roots=queue.watch.stop_owned.call_args.args[0]
            self.assertEqual([r['pid'] for r in roots],[1,2,3])
            self.assertTrue(queue.paused)

class OmniProExecutionOrderTests(unittest.TestCase):
    def test_shuffle_preserves_ids_and_existing_worker_ownership(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'scripts/shuffle_omnipro_queue.py'
        spec=importlib.util.spec_from_file_location('shuffle_order',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        order,shards=module.shuffled_order(27,2,20261031)
        self.assertEqual(sorted(order),list(range(27)))
        self.assertEqual([order[rank::2] for rank in range(2)],shards)
        self.assertEqual([set(s) for s in shards],[set(range(rank,27,2)) for rank in range(2)])
        self.assertEqual(module.shuffled_order(27,2,20261031),(order,shards))
        self.assertNotEqual(order,list(range(27)))
        self.assertNotEqual(module.shuffled_order(27,2,20261032)[0],order)
