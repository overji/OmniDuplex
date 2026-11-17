"""Temporary TP-full-credit scoring must never require an LLM or Judge GPU."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import copy
import json

import pytest
import torch

from train.src.infrastructure.config import load_config, ROOT
from train.src.infrastructure.judge import TPFullCredit
from train.src.domain.rewards import judged_reward
from train.src.training.grpo_candidates import reward_view, score_candidates
from train.tests.test_local_grpo import sample, trajectory, CONFIG, LocalEngine
from train.tests.test_grpo_judge import sample as agent_sample


TEMPLATE = ROOT/'train/configs/minicpmo_full_grpo_tp_full_credit_dp4.json'


def evaluate(qa, t, mode='local_turn_judge'):
    c = dict(CONFIG, reward_backend='tp_full_credit', reward_mode=mode)
    views = [reward_view(t)]
    packet = score_candidates(TPFullCredit(), qa, views, c)
    assert packet['ok']
    return packet['rewards'][0]


@pytest.mark.parametrize('mode',['local_turn_judge','confusion_judge'])
@pytest.mark.parametrize('interaction',['rtd','pr'])
def test_tp_maximum_keeps_duplicate_fp_fn_tn_and_boundary_matching(mode, interaction):
    qa = sample(); qa['mode'] = interaction
    t = trajectory([(2,2,'Arbitrary unrelated answer.',2),
                    (3,3,'Duplicate in the same window.',2), (9,9,'Late.',2)])
    result = evaluate(qa,t,mode)
    assert (result['tp'],result['fp'],result['fn']) == (1,2,1)
    assert result['content'] == .5  # One hit among two windows, quality exactly 1.
    answers={i:dict(status='complete',response_score=2,call_score=2) for i in range(2)}
    expected=judged_reward(qa,reward_view(t),answers,dict(CONFIG,reward_mode=mode,reward_backend='llm'))
    assert result == expected


@pytest.mark.parametrize('text',['Done.', 'Done.<|task_start|>', '',
                               'Done.<|task_start|>Unrelated instruction.'])
def test_agent_tp_has_full_quality_even_without_valid_semantics_but_keeps_format_penalty(text):
    qa = agent_sample(True);t = trajectory([(2,2,text,2)])
    result=evaluate(qa,t)
    reply=next(u for u in result['local_credit']['units'] if u['kind']=='reply')
    assert result['tp']==1 and result['content']==1 and result['call']==1
    assert reply['quality']==1
    assert reply['base_advantage']==pytest.approx(1-.1*result['invalid_calls'])


def test_control_only_attempt_is_still_fp_and_all_silent_still_gets_fn():
    qa=sample(((2,3),))
    empty=evaluate(qa,trajectory([(2,2,None,1)]))
    assert (empty['tp'],empty['fp'],empty['fn'])==(0,1,1)
    silent=evaluate(qa,trajectory())
    assert (silent['tp'],silent['fp'],silent['fn'])==(0,0,1)
    assert silent['total']==pytest.approx(-.9)


def test_scorer_never_reads_model_metadata_or_opens_http():
    qa=sample();t=trajectory([(2,2,'Anything.',2)])
    before=copy.deepcopy(t)
    with patch('train.src.infrastructure.judge.identity',side_effect=AssertionError('metadata')), \
         patch('urllib.request.build_opener',side_effect=AssertionError('HTTP')), \
         patch('urllib.request.urlopen',side_effect=AssertionError('HTTP')):
        scorer=TPFullCredit();scorer.ready();answers=scorer.score_group(qa,[reward_view(t)])
    assert answers[0][0]['response_score']==2 and answers[0][1]['response_score']==0
    assert scorer.snapshot()['requests']==0 and scorer.snapshot()['score_requests']==1
    assert scorer.identity['uses_llm'] is False and before==t


def test_template_needs_no_judge_and_rejects_invalid_reward_config(tmp_path):
    c=load_config(TEMPLATE)
    assert c['reward_backend']=='tp_full_credit' and 'judge' not in c
    assert load_config(ROOT/'train/configs/minicpmo_full_grpo_confusion_judge_dp4.json')['reward_backend']=='llm'
    path=tmp_path/'config.json'
    bad=copy.deepcopy(json.loads(TEMPLATE.read_text()));bad['reward_weights']['fp']=-1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='reward weight'):load_config(path)
    bad=copy.deepcopy(json.loads(TEMPLATE.read_text()));bad['reward_mode']='legacy'
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='requires confusion_judge'):load_config(path)


def test_checkpoint_identity_does_not_read_judge_and_differs_from_llm(tmp_path):
    from train.src.infrastructure.checkpoints import fingerprints
    for n in ('train.json','validation.json','manifest.json'):(tmp_path/n).write_text('{}')
    c=load_config(TEMPLATE)
    with patch('benchmark.data.agent_token.base_identity',return_value={}), \
         patch('train.src.infrastructure.tokens.load_training_tokenizer',return_value=None), \
         patch('train.src.infrastructure.checkpoints.tokenizer_metadata',return_value={}), \
         patch('train.src.infrastructure.judge.identity',side_effect=AssertionError('No Judge')):
        temporary=fingerprints(tmp_path,c)
    assert temporary['judge']['uses_llm'] is False
    with patch('benchmark.data.agent_token.base_identity',return_value={}), \
         patch('train.src.infrastructure.tokens.load_training_tokenizer',return_value=None), \
         patch('train.src.infrastructure.checkpoints.tokenizer_metadata',return_value={}), \
         patch('train.src.infrastructure.judge.identity',return_value={'uses_llm':True}):
        semantic=fingerprints(tmp_path,dict(c,reward_backend='llm',judge={}))
    assert temporary != semantic


def test_cpu_trainer_updates_with_no_judge_and_reports_actual_backend(tmp_path):
    from train.src.training.grpo_judged import run
    c=load_config(TEMPLATE);c.update(epochs=1,save_every_steps=0,grpo_activation_offload=False,
        grpo_candidate_parallel=False,grpo_precheck_replay=False)
    records=[dict(sample_index=i,sample=sample(),x=.1) for i in range(2)]
    engine=LocalEngine();before=engine.parameter.detach().clone();prefix='train.src.training.grpo_judged.'
    with ExitStack() as stack:
        stack.enter_context(patch(prefix+'Judge',side_effect=AssertionError('LLM Judge constructed')))
        stack.enter_context(patch(prefix+'rollout',side_effect=lambda *a:trajectory()))
        stack.enter_context(patch(prefix+'Clip',return_value=SimpleNamespace(horizon=10)))
        stack.enter_context(patch(prefix+'epoch_slots',return_value=[0,1]))
        stack.enter_context(patch(prefix+'broadcast',side_effect=lambda packet,rank:packet))
        stack.enter_context(patch(prefix+'all_true',side_effect=lambda ok,device:ok))
        for name in ('torch.cuda.reset_peak_memory_stats','torch.cuda.max_memory_allocated',
                     'torch.distributed.barrier','torch.distributed.destroy_process_group','save'):
            stack.enter_context(patch(prefix+name,return_value=0))
        run(engine,None,None,records,SimpleNamespace(output=tmp_path,profile=False,max_steps=1),
            c,torch.device('cpu'),0,1,{},MagicMock(),0.)
    report=json.loads((tmp_path/'memory_report.rank0.json').read_text())
    assert report['reward_backend']=='tp_full_credit' and report['optimizer_steps']==1
    assert engine.parameter.item()!=before.item()
    row=json.loads((tmp_path/'metrics.rank0.jsonl').read_text().splitlines()[-1])
    assert row['judge_requests']==0 and row['judge_failed_groups']==0
