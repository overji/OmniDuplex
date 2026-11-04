import importlib.util
import json
from pathlib import Path
import pytest


def test_report_normalizes_each_score_and_keeps_unmatched_zero(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/run_local_content_judge.py'
    spec=importlib.util.spec_from_file_location('local_content_judge',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'judge/base';out.mkdir(parents=True)
    rows=[dict(sample_index=0,response_index=0,status='complete',event_index=0,mode='agent',response_score=2,call_score=1,call_applicable=True),
          dict(sample_index=1,response_index=0,status='complete',event_index=0,mode='pr',response_score=1,call_score=None,call_applicable=False),
          dict(sample_index=1,response_index=1,status='complete',event_index=None,mode='pr',response_score=0,call_score=None,call_applicable=False)]
    (out/'judge_metrics.json').write_text('{}')
    (out/'judgments.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    module.normalized_report(dict(output=str(tmp_path),models=[dict(label='base')]))
    g=json.loads((tmp_path/'normalized_summary.json').read_text())['models']['base']['overall']
    assert g['response_mean_matched']==.75
    assert g['response_mean_all']==.5
    assert g['task_mean_all']==pytest.approx(1.25/3)


def test_local_scorer_uses_actual_run_schema_for_agent_references(tmp_path,monkeypatch):
    from benchmark.evaluation.omnipro import score_run
    sample=dict(question='Check this.',question_time=0,mode='agent',task='grounding_agent',video='v.mp4',
        available_tools=['search'],responses=[dict(role='assistant',window_start=1,window_end=2,response='Checking.',
            tool_calls=[dict(function=dict(name='agent_call',arguments=dict(instruction='Find the address',capability='search')))])])
    data=tmp_path/'data.json';data.write_text(json.dumps([sample]));original=data.read_bytes()
    run=tmp_path/'existing_run';run.mkdir();(run/'run_config.json').write_text(json.dumps(dict(agent_call_schema='task-start')))
    event=dict(onset_seconds=1,text='Checking.',tool_calls=[dict(name='task_start',arguments=dict(instruction='Find the address'))])
    prediction=dict(sample_index=0,status='complete',task=sample['task'],video=sample['video'],question=sample['question'],events=[event])
    (run/'predictions.jsonl').write_text(json.dumps(prediction)+'\n')
    prompt=tmp_path/'prompt.txt';prompt.write_text('Judge.')
    monkeypatch.setattr('train.src.infrastructure.judge.identity',lambda settings:{'model':'test'})
    captured=[]
    def evaluate(row,*args):
        captured.append(row)
        return {k:v for k,v in dict(row,status='complete',response_score=2,call_score=2).items() if k!='payload'}
    monkeypatch.setattr('benchmark.evaluation.omnipro.evaluate',evaluate)
    cfg=dict(output=str(tmp_path),data=str(data),run_paths=dict(sft=str(run)),judge=dict(system_prompt=str(prompt),model='test',
        base_url='http://127.0.0.1:1/v1',timeout=1,retries=0,concurrency=1,max_tokens=512))
    assert score_run(cfg,'sft')=='complete'
    payload=captured[0]['payload']
    assert payload['judge_call']
    assert payload['reference']['tool_calls'][0]['function']==dict(name='task_start',arguments=dict(instruction='Find the address'))
    assert data.read_bytes()==original
