"""Scored-F1 must penalize excess output without changing existing matching."""
import pytest
from benchmark.evaluation.scored_f1 import scored_metrics,normalized_quality,judged_metrics
from benchmark.evaluation.judge import summarize


def test_same_quality_excess_outputs_reduce_score():
    a=scored_metrics(8,10,10);b=scored_metrics(8,10,100)
    assert a['scored_f1']==.8
    assert b['scored_f1']==pytest.approx(16/110)
    assert b['scored_recall']==a['scored_recall']==.8
    assert scored_metrics(0,10,0)['scored_f1']==0
    assert scored_metrics(10,10,10)['scored_f1']==1
    assert scored_metrics(0,0,0)['scored_f1'] is None
    assert scored_metrics(None,10,10)['scored_f1'] is None
    assert scored_metrics(1,10,None)['scored_f1'] is None
    with pytest.raises(ValueError):scored_metrics(11,10,10)


def test_agent_normalization_and_failed_judges():
    row=dict(status='complete',response_score=2,call_score=1,call_applicable=True)
    assert normalized_quality(row)==.75
    assert judged_metrics([row],1,2)['scored_f1']==.5
    assert judged_metrics([dict(status='failed')],1,2)['scored_f1'] is None


def test_realtime_summary_counts_unmatched_output_and_failed_judge():
    row=dict(status='complete',sample_index=0,response_index=0,mode='pr',ability='grounding',sample_type=None,
             event_index=0,response_score=2,call_score=None,call_applicable=False)
    population={'overall':dict(prediction_count=3)}
    metrics=summarize([row],[],1,1,population)
    assert metrics['groups']['overall']['scored_f1']==.5
    row['status']='failed';row['response_score']=None
    assert summarize([row],[],1,1,population)['groups']['overall']['scored_f1'] is None


def test_portable_bundle_preserves_denominators_and_agent_quality():
    from benchmark.tests.test_judge_bundle import bundle_fixture
    from benchmark.evaluation.judge_bundle import build_result
    from benchmark.evaluation.judge_package import request_key
    b=bundle_fixture()
    for run in b['runs']:
        run['scored_f1_population']={'overall':dict(reference_count=1,prediction_count=3)}
    cache={request_key(t,b):dict(status='complete',response_score=2,call_score=None,response_reason='ok',
        call_reason='N/A',error_types=[],usage_attempts=[]) for t in b['tasks'] if t['api_required']}
    result=build_result(b,{},cache)
    assert result['summary']['model_a']['overall']['scored_f1']==.5
    assert result['summary']['agent']['overall']['scored_f1']==.25
    assert result['summary']['miss']['overall']['scored_f1']==0


def test_offline_recompute_uses_real_output_count_without_touching_sources(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    sample=dict(task='grounding_pr',mode='pr',ability='grounding',video='v.mp4',question='Notify me.',question_time=0,
        responses=[dict(role='assistant',window_start=1,window_end=2,response='Here.')])
    run=tmp_path/'run';run.mkdir();data=tmp_path/'data.json';data.write_text(json.dumps([sample]))
    pred=dict(sample_index=0,status='complete',task=sample['task'],video='v.mp4',question=sample['question'],
              events=[dict(onset_seconds=t,text='Here.') for t in [1,3,4]])
    source=run/'predictions.jsonl';source.write_text(json.dumps(pred)+'\n')
    judgments=tmp_path/'judgments.jsonl';judgments.write_text(json.dumps(dict(status='complete',sample_index=0,
        response_index=0,mode='pr',ability='grounding',sample_type=None,event_index=0,call_applicable=False,
        response_score=2,call_score=None))+'\n')
    before=[p.read_bytes() for p in [data,source,judgments]]
    script=Path(__file__).resolve().parents[1]/'scripts/recompute_scored_f1.py'
    output=tmp_path/'score.json'
    result=subprocess.run([sys.executable,str(script),'--mode','realtime','--data',str(data),'--run',str(run),
        '--judgments',str(judgments),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(output.read_text())['groups']['overall']['scored_f1']==.5
    assert before==[p.read_bytes() for p in [data,source,judgments]]
