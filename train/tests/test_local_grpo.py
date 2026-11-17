"""Local credit arithmetic, clipping gradients, and QA/candidate accumulation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from benchmark.data.task_text import event
from train.src.domain.local_reward import local_reward
from train.src.domain.rewards import judged_reward
from train.src.models.local_policy import local_policy_loss
from train.src.training.grpo_candidates import reward_view
from train.src.training.grpo_judged import update_batch
from train.tests.test_grpo_batch import Engine


WEIGHTS = dict(tp=1., fp=.5, fn=1., tn=.1, invalid_call_penalty=.1)
CONFIG = dict(reward_mode='local_turn_judge', reward_weights=WEIGHTS)


def sample(windows=((2, 3), (6, 7))):
    return dict(mode='rtd', task='object_counting_rtd', question='Update the count.',
        question_time=0, responses=[dict(window_start=a, window_end=b, response='Two buckets.') for a,b in windows])


def trajectory(replies=(), horizon=10):
    """Replies are (start, end, text-or-None, sampled-tokens-per-second)."""
    chunks = [dict(new_response=False, idle_wait=True, reply_ids=[], turn_open_before=False,
        ids=[0], actions=[dict(offset=0, reply_id=None, turn_open=False, old_log_prob=-.5)]) for _ in range(horizon)]
    events=[]
    for owner,(start,end,text,tokens) in enumerate(replies):
        for second in range(start,end+1):
            actions=[dict(offset=i,reply_id=owner,turn_open=second>start or i>0,old_log_prob=-.5) for i in range(tokens)]
            chunks[second-1]=dict(new_response=second==start,idle_wait=False,reply_ids=[owner],
                turn_open_before=second>start,ids=[1]*tokens+[999],actions=actions)
        if text is not None:events.append(dict(event(text,start,complete=True),reply_id=owner))
    return dict(chunks=chunks,events=events)


def score(qa,t,config=CONFIG,quality=2):
    judgments={i:dict(status='complete',response_score=quality,call_score=quality) for i in range(len(qa['responses']))}
    return judged_reward(qa,reward_view(t),judgments,config)


def with_credit(t,reward):
    return dict(t,local_credit=reward['local_credit'])


def loss(t,reward,current=None):
    size=sum(len(c['actions']) for c in t['chunks'])
    old=torch.full((size,),-.5,dtype=torch.float64)
    current=old.clone().requires_grad_() if current is None else current
    return current,local_policy_loss(current,old,with_credit(t,reward),.2)


def test_correct_repeat_and_missed_window_get_opposite_gradients():
    qa=sample();t=trajectory([(2,4,'Two buckets.',3),(5,5,'Two buckets again.',2)])
    r=score(qa,t)
    assert r['local_idle_fn']==1 and r['local_blocked_fn']==0
    assert r['fp']==1 and r['fn']==1
    assert r['total']==pytest.approx(-.25+.08)
    current,(value,ratio,info)=loss(t,r);value.backward()
    cursor=0
    for second,chunk in enumerate(t['chunks'],1):
        g=current.grad[cursor:cursor+len(chunk['actions'])];cursor+=len(g)
        if 2<=second<=4:assert (g<0).all()  # Gradient descent raises the correct reply probability.
        if second==5 or 6<=second<=7:assert (g>0).all()
    assert value.item()==pytest.approx(-r['total'])
    assert sum(info[k] for k in ('local_loss_reply','local_loss_fp','local_loss_fn','local_loss_wait'))==pytest.approx(value.item())
    assert '999' not in str(reward_view(t))


def test_same_total_all_silent_group_still_has_negative_window_credit():
    t=trajectory();r=score(sample(),t)
    assert r['total']==pytest.approx(-.9)
    assert all(score(sample(),t)['local_credit']==r['local_credit'] for _ in range(8))
    current,(value,_,_)=loss(t,r);value.backward()
    assert (current.grad[[1,2,5,6]]>0).all()
    assert (current.grad[[0,3,4,7,8,9]]<0).all()


def test_waiting_inside_successful_window_is_zero_and_question_prefix_masked():
    qa=sample(((2,4),));qa['question_time']=1
    t=trajectory([(4,5,'Two buckets.',2)])
    current,(value,_,_)=loss(t,score(qa,t));value.backward()
    assert current.grad[:3].tolist()==[0.,0.,0.]


def test_occupied_fn_charges_entire_reply_and_excludes_continuation_from_tn():
    t=trajectory([(2,8,'Two buckets.',2)]);r=score(sample(),t)
    unit=next(u for u in r['local_credit']['units'] if u['kind']=='reply')
    assert r['local_blocked_fn']==1 and r['local_idle_fn']==0
    assert unit['base_advantage']==1 and unit['advantage']==0
    assert unit['blocked_windows']==[dict(response_index=1,fraction=1.)]
    assert r['local_wait_opportunities']==3
    current,(value,_,info)=loss(t,r);value.backward()
    assert (current.grad[1:15]==0).all()
    assert info['local_loss_blocked_fn']==pytest.approx(.5)


def test_occupied_fn_split_across_replies_and_empty_attempt_penalized():
    qa=sample(((2,3),(4,5)))
    t=trajectory([(2,4,'Two buckets.',2),(5,6,None,1)])
    r=score(qa,t)
    units=[u for u in r['local_credit']['units'] if u['kind']=='reply']
    assert r['unparsed_response_starts']==1 and r['fp']==1
    assert [u['blocked_fn_advantage'] for u in units]==[-.5,-.5]
    assert [u['advantage'] for u in units]==[.5,-1.]


def test_reply_length_window_width_and_wait_duration_normalize_once():
    qa=sample(((2,3),))
    values=[]
    for tokens in (1,30):
        t=trajectory([(2,2,'Two buckets.',tokens)])
        values.append(loss(t,score(qa,t))[1][0].item())
    assert values[0]==pytest.approx(values[1])
    for windows,horizon in [(((2,3),),10),(((2,8),),20),(((2,3),),1000)]:
        t=trajectory(horizon=horizon);r=score(sample(windows),t)
        assert r['total']==pytest.approx(-.9)
        assert loss(t,r)[1][0].item()==pytest.approx(.9)


def test_clipped_objective_and_gradients_match_explicit_unit_reference():
    t=trajectory([(2,4,'Two buckets.',3),(5,5,'Again.',2)])
    r=score(sample(),t);size=sum(len(c['actions']) for c in t['chunks'])
    old=torch.full((size,),-.5,dtype=torch.float64)
    current=(old+torch.linspace(-.5,.5,size)).requires_grad_()
    value,_,_=local_policy_loss(current,old,with_credit(t,r),.2)
    other=current.detach().clone().requires_grad_();ratio=(other-old).exp();bounded=ratio.clamp(.8,1.2)
    by_reply={};idle={};offset=0
    for second,c in enumerate(t['chunks'],1):
        for j,a in enumerate(c['actions'],offset):
            if a['reply_id'] is not None:by_reply.setdefault(a['reply_id'],[]).append(j)
            elif c['idle_wait']:idle.setdefault(second,[]).append(j)
        offset+=len(c['actions'])
    expected=other.sum()*0
    for u in r['local_credit']['units']:
        groups=[by_reply[u['reply_id']]] if u['kind']=='reply' else [idle[s] for s in u['seconds']]
        terms=[-torch.minimum(ratio[g]*u['advantage'],bounded[g]*u['advantage']).mean() for g in groups]
        expected+=sum(terms)/len(terms)*u['weight']
    torch.testing.assert_close(value,expected)
    torch.testing.assert_close(torch.autograd.grad(value,current)[0],torch.autograd.grad(expected,other)[0])


def test_agent_scores_invalid_format_and_judge_failure():
    from train.tests.test_grpo_judge import sample as agent_sample
    qa=agent_sample(True)
    for text,expected in [('Checking.<|task_start|>Write a note.',1.),('Checking.<|task_start|>',-.1)]:
        t=trajectory([(2,2,text,4)])
        r=score(qa,t)
        unit=next(u for u in r['local_credit']['units'] if u['kind']=='reply')
        assert unit['advantage']==pytest.approx(expected)
    with pytest.raises(ValueError,match='Failed judge'):
        local_reward(qa,reward_view(t),{0:dict(status='failed')},CONFIG)
    with pytest.raises(ValueError,match='telemetry'):
        local_reward(sample(),dict(events=[],chunks=[dict(new_response=False,idle_wait=True)]*10),{},CONFIG)


class LocalEngine(Engine):
    def __call__(self,record,clip,trajectory):
        size=sum(len(c['actions']) for c in trajectory['chunks'])
        old=torch.full((size,),-.5)
        current=old+self.parameter*record['x']
        if 'local_credit' not in trajectory:
            return current.detach(), (current.detach()-old).abs().max()
        value,ratio,info=local_policy_loss(current,old,trajectory,.2)
        return value,dict(info,replay_max_logprob_error=0.,ratio_mean=ratio.mean().item())


def make_groups(count):
    result=[]
    for i in range(count):
        candidates=[trajectory([(2,2,'Correct.',j+1)]) if j%2==0 else trajectory() for j in range(4)]
        credits=[score(sample(),t)['local_credit'] for t in candidates]
        result.append(dict(record=dict(x=.1*(i+1)),clip=None,trajectories=candidates,local_credits=credits))
    return result


def test_local_accumulation_padding_matches_reference_and_rejects_malformed_credit():
    for count in (1,3,4):
        groups=make_groups(count);engine=LocalEngine();expected=LocalEngine()
        reference=sum(sum(expected(g['record'],None,dict(t,local_credit=c))[0]
            for t,c in zip(g['trajectories'],g['local_credits']))/4 for g in groups)/count
        grad=torch.autograd.grad(reference,expected.parameter)[0]
        with patch('train.src.training.grpo_judged.all_true',side_effect=lambda ok,device:ok):
            stats=update_batch(engine,groups,dict(CONFIG,gradient_accumulation_steps=4,
                replay_logprob_tolerance=.15,grpo_activation_offload=False),torch.device('cpu'))
        torch.testing.assert_close(engine.parameter,expected.parameter.detach()-.01*grad)
        assert engine.global_steps==1 and stats['padding_slots']==4-count
        assert stats['loss']==pytest.approx(reference.item())
    t=trajectory();r=score(sample(),t);r['local_credit']['units'][0]['seconds']=[100]
    with pytest.raises(ValueError):loss(t,r)


def test_native_rollout_ownership_cross_seconds_multiple_turns_and_forced_tokens():
    from contextlib import nullcontext
    from train.src.models.streaming import rollout
    from train.tests.helpers import ByteTokenizer
    class Duplex:
        def __init__(self):
            self.tokenizer=ByteTokenizer();self.listen_token_id=302
            self.chunk_eos_token_id=304;self.tts_bos_token_id=301
            self.chunk_terminator_token_ids={302,304};self.turn_terminator_token_ids={303}
            self.current_turn_ended=True;self.total_ids=[];self.calls=[];self.index=0
            self.decoder=SimpleNamespace(decode=lambda **kw: None,forbidden_token_ids=[304],
                reset=lambda: None,get_cache_length=lambda: len(self.total_ids))
        def prepare(self,**kw):pass
        def streaming_prefill(self,**kw):
            self.calls.append(('question' if 'text_list' in kw else 'media',self.index))
            return dict(success=True)
        def streaming_generate(self,**kw):
            plan=[[302],[301,65],[66,303,304],[301,67,303,301,68,303,304],[301,304],[304]][self.index]
            self.calls.append(('generate',self.index));self.index+=1
            end=False;listen=False
            for token in plan:
                if self.index<=self.force_listen_count:
                    actual=302
                else:
                    logits=torch.zeros(1,320)
                    actual=self.decoder.decode(logits=logits).item()
                    assert actual==token
                self.total_ids.append(actual)
                listen=actual==302
                if actual in self.chunk_terminator_token_ids:break
                self.current_turn_ended=False
                end=actual in self.turn_terminator_token_ids
                if end:self.current_turn_ended=True
            self.total_ids.extend([304,305])  # Forced tokens never enter actions.
            return dict(is_listen=listen,end_of_turn=end)
    duplex=Duplex();native=duplex.decoder.decode
    model=SimpleNamespace(eval=lambda: None,init_streaming_processor=lambda: None,audio_past_key_values=None)
    clip=SimpleNamespace(question_time=1,chunks=[{}]*6,preprocessing_cache=lambda _:nullcontext())
    duplex.processor=None
    sampled=[301,65,66,303,304,301,67,303,301,68,303,304,301,304,304]
    with patch('train.src.models.streaming.system_prompt',return_value='system'), \
         patch('train.src.models.streaming.user_request',return_value='question'), \
         patch('torch.multinomial',side_effect=[torch.tensor([x]) for x in sampled]):
        t=rollout(model,duplex,dict(sample=sample()),clip,
            dict(CONFIG,batch_prompt_prefill=False,max_speak_tokens_per_chunk=20,max_sequence_tokens=100),42)
    assert [(e['response_text'],e['reply_id']) for e in t['events']]==[('AB',0),('C',1),('D',2)]
    assert t['chunks'][0]['actions']==[]
    assert [a['reply_id'] for a in t['chunks'][2]['actions']]==[0,0,None]
    assert t['chunks'][3]['reply_ids']==[1,2]
    assert t['chunks'][4]['reply_ids']==[3] and not t['chunks'][4]['idle_wait']
    assert sum(len(c['actions']) for c in t['chunks'])==len(sampled)
    assert duplex.calls.index(('question',1))<duplex.calls.index(('media',1))<duplex.calls.index(('generate',1))
    assert duplex.decoder.decode is native


def test_real_tiny_graph_local_loss_replays_probabilities_and_backpropagates():
    from train.tests.test_grpo import ReplayGradientTests
    with patch('train.src.models.streaming.user_request',return_value='question'), \
         patch('train.src.models.streaming.system_prompt',return_value='system'):
        graph,record,clip=ReplayGradientTests().setup_graph();graph.eval()
        actions=[dict(offset=i,turn_open=i>0,reply_id=0,old_log_prob=0.) for i in range(4)]
        t=dict(chunks=[dict(ids=[301,97,300,98,303,304,305],actions=actions,idle_wait=False)])
        with torch.no_grad():current,_=graph(record,clip,t)
        for a,p in zip(actions,current.tolist()):a['old_log_prob']=p
        t['local_credit']=dict(version=1,units=[dict(kind='reply',reply_id=0,action_count=4,
            advantage=1.,base_advantage=1.,blocked_fn_advantage=0.,response_index=0,weight=1.)])
        value,info=graph(record,clip,t)
        assert info['replay_max_logprob_error']<1e-5
        assert value.item()==pytest.approx(-1.)
        value.backward()
        assert graph.model.llm.lm_head.weight.grad.abs().sum()>0


def test_trainer_updates_identical_silent_candidates_and_excludes_judge_failure(tmp_path):
    from contextlib import ExitStack
    from unittest.mock import MagicMock
    from train.src.training.grpo_judged import run
    from train.src.infrastructure.config import load_config
    config=load_config('train/configs/minicpmo_full_grpo_confusion_judge.json')
    config.update(epochs=1,save_every_steps=0,grpo_activation_offload=False,grpo_judge_pipeline_depth=2)
    class Judge:
        def __init__(self,*a):pass
        def ready(self):pass
        def snapshot(self):return dict(requests=1,cache_hits=0,failures=1,retries=0)
        def score_group(self,qa,ts):
            if qa['question']=='fail':raise TimeoutError()
            return [{} for _ in ts]
    records=[dict(sample_index=i,sample=dict(sample(),question='fail' if i==0 else 'wait'),x=.1) for i in range(3)]
    engine=LocalEngine()
    prefix='train.src.training.grpo_judged.'
    with ExitStack() as stack:
        stack.enter_context(patch(prefix+'Judge',Judge))
        stack.enter_context(patch(prefix+'rollout',side_effect=lambda *a:trajectory()))
        stack.enter_context(patch(prefix+'Clip',return_value=SimpleNamespace(horizon=10)))
        stack.enter_context(patch(prefix+'epoch_slots',return_value=[0,1,2]))
        stack.enter_context(patch(prefix+'broadcast',side_effect=lambda packet,rank:packet))
        stack.enter_context(patch(prefix+'all_true',side_effect=lambda ok,device:ok))
        for name in ('torch.cuda.reset_peak_memory_stats','torch.cuda.max_memory_allocated',
                     'torch.distributed.barrier','torch.distributed.destroy_process_group','save'):
            stack.enter_context(patch(prefix+name,return_value=0))
        run(engine,None,None,records,SimpleNamespace(output=tmp_path,profile=False,max_steps=0),
            config,torch.device('cpu'),0,4,{},MagicMock(),0.)
    import json
    report=json.loads((tmp_path/'memory_report.rank0.json').read_text())
    assert report['valid_qa_this_run']==2 and report['excluded']==1
    assert report['zero_variance_groups']==0 and engine.global_steps==1
    rows=[json.loads(x) for x in (tmp_path/'metrics.rank0.jsonl').read_text().splitlines()]
    update=next(r for r in rows if r.get('optimizer_boundary'))
    assert update['local_negative_units']>0 and update['local_idle_fn']==2
    saved=[json.loads(x) for x in (tmp_path/'trajectories.rank0.jsonl').read_text().splitlines()]
    assert all(r['reward']['local_credit']['version']==1 for r in saved)


def distributed_local_worker(rank,world,rendezvous):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',rank=rank,world_size=world,
        init_method=rendezvous,timeout=timedelta(seconds=45))
    class DistributedLocalEngine(LocalEngine):
        def step(self):
            if (self.micro_steps+1)%self.gas==0:
                torch.distributed.all_reduce(self.parameter.grad)
                self.parameter.grad.div_(world)
            super().step()
    try:
        for count in (1,3,4):
            full=make_groups(count);engine=DistributedLocalEngine();expected=LocalEngine()
            local=[dict(g,trajectories=[g['trajectories'][rank]],local_credits=[g['local_credits'][rank]]) for g in full]
            reference=sum(sum(expected(g['record'],None,dict(t,local_credit=c))[0]
                for t,c in zip(g['trajectories'],g['local_credits']))/world for g in full)/count
            grad=torch.autograd.grad(reference,expected.parameter)[0]
            config=dict(CONFIG,gradient_accumulation_steps=4,replay_logprob_tolerance=.15,
                        grpo_activation_offload=False,grpo_candidate_parallel=True)
            result=update_batch(engine,local,config,torch.device('cpu'))
            torch.testing.assert_close(engine.parameter,expected.parameter.detach()-.01*grad)
            assert result['candidate_trajectories']==count*world
            assert result['loss']==pytest.approx(reference.item())
            assert engine.global_steps==1
            summed=sum(result['local_loss_'+k] for k in ('reply','fp','fn','wait'))
            assert summed==pytest.approx(result['loss'])
    finally:
        torch.distributed.destroy_process_group()


def test_four_cpu_candidate_replicas_equal_full_candidate_reference(tmp_path):
    import torch.multiprocessing as mp
    # Spawn avoids inheriting a parent autograd thread pool.
    mp.start_processes(distributed_local_worker,args=(4,'file://'+str(tmp_path/'rendezvous')),
                       nprocs=4,join=True,start_method='spawn')


def test_unassignable_fn_is_retained_in_score_and_reported_without_fake_gradients():
    t=trajectory()
    for second in (2,3,6,7):
        t['chunks'][second-1]['actions']=[]
    r=score(sample(),t)
    assert r['local_unassigned_fn']==2
    assert r['local_credit']['unassigned_windows']==[0,1]
    assert r['total']==pytest.approx(-.9)
    assert all(u['kind']=='wait' for u in r['local_credit']['units'])
    assert loss(t,r)[1][0].item()==pytest.approx(-.1)


def test_local_checkpoint_identity_pins_mode_and_reward_aggregation_sources(tmp_path):
    import hashlib
    from train.src.infrastructure.checkpoints import fingerprints
    from train.src.infrastructure.config import load_config, ROOT
    for name in ('train.json','validation.json','manifest.json'):
        (tmp_path/name).write_text('{}')
    config=load_config('train/configs/minicpmo_full_grpo_confusion_judge.json')
    with patch('benchmark.data.agent_token.base_identity',return_value={}), \
         patch('train.src.infrastructure.tokens.load_training_tokenizer',return_value=None), \
         patch('train.src.infrastructure.checkpoints.tokenizer_metadata',return_value={}), \
         patch('train.src.infrastructure.judge.identity',return_value={}):
        current=fingerprints(tmp_path,config)
        previous=fingerprints(tmp_path,dict(config,reward_mode='confusion_judge'))
    assert current['grpo_objective']=='local_turn_ablation_v2'
    assert current!=previous
    for relative in ('domain/local_reward.py','models/local_policy.py','models/graph.py'):
        expected=hashlib.sha256((ROOT/'train/src'/relative).read_bytes()).hexdigest()
        assert current['grpo_reward_code'][relative]==expected
