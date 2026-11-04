import copy
import json
from data_quality.core import CHECKS,frame_times,static_findings,validate_judgment
import pytest


def test_sparse_selection_is_causal_and_contains_window_boundary():
    times=frame_times(10,8,24,8)
    assert 2<=len(times)<=24
    assert min(times)>=0 and max(times)<10
    assert any(abs(t-8)<.01 for t in times)
    assert min(times)<2


def test_future_or_unsupported_issue_is_not_a_valid_judgment():
    value=dict(checks={k:'pass' for k in CHECKS},findings=[],summary='Grounded.')
    assert validate_judgment(value,10)['verdict']=='pass'
    value['checks']['trigger_support']='issue'
    with pytest.raises(ValueError,match='cited evidence'):validate_judgment(value,10)
    value['findings']=[dict(check='trigger_support',severity='error',reason='Mismatch',evidence=[dict(time_seconds=11,observation='Later scene')])]
    with pytest.raises(ValueError,match='future'):validate_judgment(value,10)
    value['findings'][0]['evidence'][0]['time_seconds']=9
    assert validate_judgment(value,10)['verdict']=='issue'


def test_window_tail_overrun_is_warning_not_unreachable():
    sample=dict(question='Alert me.',question_time='00:00',mode='pr',responses=[dict(response='Now.',window_start='00:09',window_end='00:12')])
    assert static_findings(sample,10)==[dict(code='window_tail_beyond_video',severity='warning',response_index=0)]
    sample['responses'][0]['window_start']='00:11'
    assert static_findings(sample,10)[0]['code']=='unreachable_window'


def test_prepare_and_report_cover_each_response_once_without_editing_source(tmp_path):
    from data_quality.runner import prepare,report
    (tmp_path/'v.mp4').touch()
    samples=[dict(video='v.mp4',mode='pr',task='grounding_pr',question='Notify me.',question_time='00:00',
        responses=[dict(response='Now.',window_start='00:02',window_end='00:04'),dict(response='Again.',window_start='00:06',window_end='00:08')])]
    source=tmp_path/'source.json';source.write_text(json.dumps(samples));original=source.read_bytes()
    meta=tmp_path/'metadata.json';meta.write_text(json.dumps(dict(videos=[dict(file='v.mp4',duration_seconds=10)])))
    c=dict(output=str(tmp_path/'audit'),model_path='external-model',gpus=list(map(str,range(8))),media_workers=1,
        datasets=[dict(name='test',data=str(source),videos_dir=str(tmp_path),metadata=str(meta))],report_only=True)
    prepare(c)
    manifest=json.loads((tmp_path/'audit/manifest.json').read_text());jobs=[r for shard in manifest['shards'] for r in shard]
    assert len(jobs)==len({r['id'] for r in jobs})==2
    assert sorted(r['prefix_end_seconds'] for r in jobs)==[4,8]
    for rank,shard in enumerate(manifest['shards']):
        d=tmp_path/'audit'/f'worker_{rank}';d.mkdir()
        (d/'judgments.jsonl').write_text(''.join(json.dumps(dict(j,status='failed',error='test error'))+'\n' for j in shard))
    done,total=report(c)
    assert done==total==2
    summary=json.loads((tmp_path/'audit/summary.json').read_text())
    assert summary['datasets']['test']['counts']=={'failed':2}
    assert source.read_bytes()==original


def test_four_gpu_execution_visits_all_eight_original_shards(tmp_path,monkeypatch):
    from data_quality import runner
    launches=[]
    class Child:
        def __init__(self,argv,**kwargs):
            self.pid=100+len(launches)
            launches.append((int(argv[-1]),kwargs['env']['CUDA_VISIBLE_DEVICES']))
        def poll(self):return 0
    config=dict(output=str(tmp_path),gpus=list(map(str,range(8))),wait_for_previous={},
        gpu_release_timeout_seconds=1,poll_seconds=1,python='python',config_path='config.json',minimum_success_fraction=.9)
    (tmp_path/'summary.json').write_text(json.dumps(dict(datasets={'test':dict(counts={'pass':8})})))
    monkeypatch.setattr('train.scripts.sft_then_split_bench.previous_ready',lambda _:True)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:'\n'.join(f'{i}, {0 if i<4 else 70000}' for i in range(8)))
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    monkeypatch.setattr(runner,'report',lambda _:(8,8))
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(runner.signal,'signal',lambda *a:None)
    runner.run(config,['0','1','2','3'])
    assert launches==[(0,'0'),(1,'1'),(2,'2'),(3,'3'),(4,'0'),(5,'1'),(6,'2'),(7,'3')]
    assert json.loads((tmp_path/'status.json').read_text())['stage']=='complete'


def test_deferred_duplicate_controller_preserves_current_owner(tmp_path):
    import fcntl
    from data_quality.runner import run
    status=tmp_path/'status.json';status.write_text('{"pid":123,"stage":"gemma4_audit_running"}')
    before=status.read_bytes()
    with (tmp_path/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(dict(output=str(tmp_path),gpus=['0']))
    assert status.read_bytes()==before
