"""Live summaries use completed identical inputs and do not invent missing scores."""
import json
from pathlib import Path
import pytest
from benchmark.evaluation.live_report import render


def fixture(tmp_path):
    samples=[dict(mode='rtd',ability='object_counting',task='object_counting_rtd',video='v.mp4',question='count',question_time=0,
                  responses=[dict(window_start=2,window_end=3,response='one')])]
    data=tmp_path/'data.json';data.write_text(json.dumps(samples))
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    r=dict(sample_index=0,status='complete',events=[dict(onset_seconds=2,text='one')],observed_seconds=10,system_prompt_sha256='same')
    (old/'predictions.jsonl').write_text(json.dumps(r)+'\n')
    return dict(title='test',data=str(data),output=str(tmp_path/'live.md'),models=[dict(name='old',run=str(old)),dict(name='new',run=str(new))]),r,new


def test_waiting_run_is_na_not_zero_and_partial_json_tail_is_ignored(tmp_path):
    c,r,new=fixture(tmp_path);assert render(c)==0
    assert '| new | N/A |' in Path(c['output']).read_text()
    (new/'predictions.jsonl').write_text(json.dumps(r)+'\n'+json.dumps(dict(sample_index=99)))
    assert render(c)==1
    x=json.loads(Path(c['output']).with_suffix('.json').read_text())
    assert x['paired']['new']['groups']['overall']['timing_f1']==1.


def test_prompt_mismatch_and_changed_references_are_not_paired(tmp_path):
    c,r,new=fixture(tmp_path);r['system_prompt_sha256']='different';(new/'predictions.jsonl').write_text(json.dumps(r)+'\n')
    assert render(c)==0
    other=tmp_path/'other.json';other.write_text('[]');c['models'][0]['data']=str(other)
    with pytest.raises(ValueError,match='reference data differs'):render(c)


def test_index_contains_scores_and_signed_pp_on_same_subset(tmp_path):
    from benchmark.evaluation.live_report import comparison_section,change
    c,r,new=fixture(tmp_path)
    # Baseline misses this window; latest hits it.
    old=Path(c['models'][0]['run'])/'predictions.jsonl'
    wrong={**r,'events':[dict(onset_seconds=7,text='late')]}
    old.write_text(json.dumps(wrong)+'\n')
    (new/'predictions.jsonl').write_text(json.dumps(r)+'\n')
    render(c);data=json.loads(Path(c['output']).with_suffix('.json').read_text())
    text='\n'.join(comparison_section(c,data))
    assert '| 总 F1 | 0.00% | **100.00%** | ↑ +100.00 pp |' in text
    assert 'mode:rtd' in text and 'task:object_counting_rtd' in text and 'length:≤60s' in text
    assert change(.1,.2)=='↓ -10.00 pp'
    assert change(0,0)=='→ 0.00 pp'
    assert change(None,.1)=='N/A'


def test_index_waiting_candidate_preserves_missing_scores(tmp_path):
    from benchmark.evaluation.live_report import comparison_section
    c,_,_=fixture(tmp_path);render(c)
    data=json.loads(Path(c['output']).with_suffix('.json').read_text())
    text='\n'.join(comparison_section(c,data))
    assert '| 总 F1 | N/A | N/A | N/A |' in text


def test_final_refresh_does_not_fail_completed_queue_when_watcher_owns_lock(tmp_path):
    import fcntl
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts/watch_live_results.py'
    output=tmp_path/'live.md'
    config=tmp_path/'reports.json'
    config.write_text(json.dumps(dict(output=str(output),interval_seconds=60,reports=[])))
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=subprocess.run([sys.executable,str(script),'--config',str(config),'--once'],
                              capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert 'active watcher' in result.stdout
    assert not output.exists()  # Never overwrite the active writer's report.
