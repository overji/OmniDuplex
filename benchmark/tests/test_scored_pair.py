"""Scored-Pair gates content credit on both decisions, including portable scoring."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.evaluation.scored_pair import probe_population, judged_pair_metrics, scored_pair_metrics


def pair(pid='a', mode='pr'):
    return [dict(pair_id=pid, probe_id=pid+':'+side, side=side, mode=mode,
                 ability='grounding') for side in ('pre', 'post')]


def prediction(pid, text):
    return dict(probe_id=pid, status='complete', output_protocol='yes_no_v1', raw_response=text)


def judgment(pid='a:post', response=2, agent=False, call=None):
    return dict(probe_id=pid, status='complete', mode='agent' if agent else 'pr',
                ability='grounding', response_score=response, call_applicable=agent,
                call_score=call, task_score=(response+call)/2 if agent else response)


@pytest.mark.parametrize('pre,post,expected', [
    ('NO','YES',1), ('YES','YES',0), ('NO','NO',0), ('YES','NO',0),
    ('maybe','YES',0), ('NO','maybe',0)])
def test_only_both_correct_decisions_receive_quality(pre, post, expected):
    p=probe_population(pair(), [prediction('a:pre',pre),prediction('a:post',post)])['overall']
    result=judged_pair_metrics([judgment()],p)
    assert result['scored_pair']==expected
    assert result['scored_pair']<=p['correct_pair_count']/p['pair_count']
    assert 'scored_f1' not in result


def test_quality_normalization_and_required_judges_only():
    p=probe_population(pair(),[prediction('a:pre','NO'),prediction('a:post','YES')])['overall']
    assert judged_pair_metrics([judgment(response=1)],p)['scored_pair']==.5
    assert judged_pair_metrics([judgment(response=2,agent=True,call=1)],p)['scored_pair']==.75
    assert judged_pair_metrics([],p)['scored_pair'] is None
    assert judged_pair_metrics([dict(probe_id='a:post',status='failed')],p)['scored_pair'] is None
    p=probe_population(pair(),[prediction('a:pre','YES'),prediction('a:post','YES')])['overall']
    assert judged_pair_metrics([dict(probe_id='a:post',status='failed')],p)['scored_pair']==0
    assert scored_pair_metrics(0,0,0)['scored_pair'] is None
    with pytest.raises(ValueError):scored_pair_metrics(1,2,0)
    with pytest.raises(ValueError):scored_pair_metrics(0,1,2)


def test_complete_pairs_grouping_and_micro_denominator():
    manifest=pair('a')+pair('b','agent')+pair('missing')
    predictions=[prediction('a:pre','NO'),prediction('a:post','YES'),
                 prediction('b:pre','YES'),prediction('b:post','YES'),prediction('missing:post','YES')]
    pop=probe_population(manifest,predictions)
    assert pop['overall']['pair_count']==2
    assert pop['overall']['correct_pair_count']==1
    rows=[judgment(),judgment('b:post',agent=True,call=2)]
    assert judged_pair_metrics(rows,pop['overall'])['scored_pair']==.5
    assert judged_pair_metrics(rows,pop['mode:pr'])['scored_pair']==1
    assert judged_pair_metrics(rows,pop['mode:agent'])['scored_pair']==0
    with pytest.raises(ValueError,match='Duplicate'):
        probe_population(manifest,predictions+[predictions[0]])


def test_normal_summary_matches_portable_and_legacy_requires_reexport():
    from benchmark.evaluation.probe_judge import summarize
    from benchmark.evaluation.judge_bundle import build_result
    from benchmark.evaluation.judge_package import request_key
    from benchmark.tests.test_judge_bundle import bundle_fixture
    b=bundle_fixture()
    population=dict(pair_count=2,correct_pair_count=1,correct_post_probe_ids=['a:post'])
    for run in b['runs']:
        run['mode']='probe';run['scored_pair_population']={'overall':population}
    for task in b['tasks']:task['row']['probe_id']='a:post'
    cache={request_key(t,b):dict(status='complete',response_score=2,call_score=None,
           error_types=[],usage_attempts=[]) for t in b['tasks'] if t['api_required']}
    result=build_result(b,{},cache)
    normal=summarize([judgment()],{},4,4,{'overall':population})
    assert result['summary']['model_a']['overall']['scored_pair']==normal['groups']['overall']['scored_pair']==.5
    assert result['summary']['agent']['overall']['scored_pair']==.25
    silent=dict(bundle_id='test',tasks=[],runs=[dict(run_id='silent',mode='probe',
        scored_pair_population={'overall':dict(pair_count=3,correct_pair_count=0,correct_post_probe_ids=[])})])
    assert build_result(silent,{},{})['summary']['silent']['overall']['scored_pair']==0
    silent['runs'][0].pop('scored_pair_population')
    silent['runs'][0]['scored_f1_population']={'overall':dict(reference_count=3,prediction_count=0,tp_probe_ids=[])}
    legacy=build_result(silent,{},{})['summary']['silent']['overall']
    assert legacy['scored_pair'] is None and 'scored_pair_unavailable_reason' in legacy
    assert 'scored_f1' not in legacy


def test_probe_offline_recompute_preserves_sources(tmp_path):
    from benchmark.tests.test_probe_judge import sample, prediction as predict
    from benchmark.probe.data import sample_probes, file_hash
    from benchmark.evaluation.probe_judge import prepare_rows
    s=sample();s['responses']=s['responses'][:1]
    data=tmp_path/'data.json';data.write_text(json.dumps([s]))
    manifest,_=sample_probes(s,0,file_hash(data),Path('/v.mp4'),
        dict(duration=30,has_audio=True,source=dict(path='/v.mp4',size=1,mtime_ns=1)))
    mp=tmp_path/'manifest.jsonl';mp.write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    predictions=[predict(r,'<silent>' if r['side']=='pre' else 'The cup moved.') for r in manifest]
    run=tmp_path/'run';run.mkdir();pp=run/'probe_predictions.jsonl'
    pp.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    rows,_,_=prepare_rows([s],manifest,predictions)
    rows[0].update(status='complete',response_score=1,call_score=None,task_score=1)
    jp=tmp_path/'judgments.jsonl';jp.write_text(json.dumps(rows[0])+'\n')
    sources=[data,mp,pp,jp];before=[p.read_bytes() for p in sources]
    output=tmp_path/'score.json';script=Path(__file__).resolve().parents[1]/'scripts/recompute_scored_f1.py'
    process=subprocess.run([sys.executable,str(script),'--mode','probe','--data',str(data),
        '--manifest',str(mp),'--run',str(run),'--judgments',str(jp),'--output',str(output)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    result=json.loads(output.read_text())
    assert result['content_metric']=='scored_pair_v1'
    assert result['groups']['overall']['scored_pair']==.5
    assert [p.read_bytes() for p in sources]==before
