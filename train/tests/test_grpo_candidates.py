"""CPU checks: candidate ownership, judge overlap, on-policy boundaries and gradients."""
import json
from datetime import timedelta
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import torch.multiprocessing as mp

from train.src.training.grpo_candidates import candidate_ids, reward_view, gather_candidates, JudgePipeline
from train.src.training.grpo_judged import update_batch
from train.src.infrastructure.config import ROOT, load_config
from train.tests.test_grpo_batch import Engine, groups


def test_candidate_ownership_and_compact_reward_input():
    config=dict(grpo_candidate_parallel=True,group_size=4)
    assert [candidate_ids(config,r,4) for r in range(4)]==[[0],[1],[2],[3]]
    with pytest.raises(ValueError):
        candidate_ids(config,0,3)
    t=dict(events=[],chunks=[dict(new_response=False,idle_wait=True,actions=['large'],ids=[123])])
    assert reward_view(t)==dict(events=[],chunks=[dict(new_response=False,idle_wait=True)])


def test_judge_overlaps_sampling_and_only_runs_on_rank_zero():
    entered,finish=threading.Event(),threading.Event()
    def score(*args):
        entered.set()
        assert finish.wait(5)
        return dict(ok=True,rewards=[])
    with patch('train.src.training.grpo_candidates.score_candidates',side_effect=score):
        p=JudgePipeline(None,{},0)
        try:
            p.submit(dict(record=dict(sample={})),[])
            assert entered.wait(5)
            # The foreground can sample the next group while the judge is blocked.
            p.submit(dict(record=dict(sample={})),[])
            assert len(p.queue)==2
            finish.set()
            assert p.pop()[1]['ok']
            assert p.pop()[1]['ok']
        finally:
            finish.set();p.close()
        peer=JudgePipeline(None,{},1)
        peer.submit(dict(record=dict(sample={})),None)
        assert peer.pop()[1] is None
        peer.close()


def test_optimized_config_rejects_zero3_and_invalid_pipeline(tmp_path):
    c=load_config(ROOT/'train/configs/minicpmo_full_grpo_confusion_judge.json')
    c.update(grpo_candidate_parallel=True,deepspeed_config=str(ROOT/'train/configs/zero2_cpu.json'),
             grpo_precheck_replay=False,grpo_judge_pipeline_depth=2)
    p=tmp_path/'config.json'
    p.write_text(json.dumps(c));load_config(p)
    for bad in ({'deepspeed_config':str(ROOT/'train/configs/zero3_cpu.json')},
                {'grpo_judge_pipeline_depth':5},{'grpo_candidate_parallel':1}):
        p.write_text(json.dumps({**c,**bad}))
        with pytest.raises(ValueError):load_config(p)


def test_replay_trajectory_error_reaches_collective_before_aborting():
    from train.src.domain.trajectory import TrajectoryError
    class BadEngine(Engine):
        def __call__(self,*args):raise TrajectoryError('invalid_local_replay')
    engine=BadEngine()
    with patch('train.src.training.grpo_judged.all_true',side_effect=lambda ok,device:ok) as collective:
        with pytest.raises(RuntimeError,match='replay mismatch'):
            update_batch(engine,groups(1),dict(gradient_accumulation_steps=4,
                         grpo_activation_offload=False,replay_logprob_tolerance=.15),torch.device('cpu'))
        collective.assert_called_once_with(False,torch.device('cpu'))
    assert engine.global_steps==0


def distributed_worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',rank=rank,world_size=world,
                                        init_method=rendezvous,timeout=timedelta(seconds=45))
    class DistributedEngine(Engine):
        def step(self):
            if (self.micro_steps+1)%self.gas==0:
                torch.distributed.all_reduce(self.parameter.grad)
                self.parameter.grad.div_(world)
            super().step()
    try:
        t=dict(events=[dict(candidate=rank)],chunks=[dict(new_response=False,idle_wait=True)])
        views=gather_candidates([t],True,rank,world)
        if rank==0:assert [v['events'][0]['candidate'] for v in views]==list(range(world))
        for count in (1,3,4):
            engine=DistributedEngine()
            full=groups(count)
            local=[{**g,'trajectories':[g['trajectories'][rank]],'advantages':[g['advantages'][rank]]} for g in full]
            config=dict(gradient_accumulation_steps=4,replay_logprob_tolerance=.15,
                        grpo_activation_offload=False,grpo_candidate_parallel=True)
            result=update_batch(engine,local,config,torch.device('cpu'))
            parameter=torch.tensor(.7,requires_grad=True)
            reference=sum(sum((parameter*g['record']['x']-t['target'])**2*a
                              for t,a in zip(g['trajectories'],g['advantages']))/world for g in full)/count
            reference.backward()
            torch.testing.assert_close(engine.parameter,parameter.detach()-.01*parameter.grad)
            assert result['candidate_trajectories']==count*world
            assert engine.global_steps==1
            assert abs(result['loss']-reference.item())<1e-5
        # Exercise the complete asynchronous trainer with real collectives too.
        from contextlib import ExitStack
        from unittest.mock import MagicMock
        from train.src.training.grpo_judged import run
        from train.tests.test_grpo_judge import sample, trajectory
        class FakeJudge:
            def __init__(self, settings):pass
            def ready(self):pass
            def snapshot(self):return dict(requests=1,cache_hits=0,failures=0,retries=0)
            def score_group(self, qa, trajectories):
                return [{0:dict(status='complete',response_score=s,call_score=None)}
                        for s in (0,.5,1.5,2)]
        engine=DistributedEngine()
        records=[dict(sample_index=i,sample=sample(),x=i+1) for i in range(6)]
        def fake_rollout(model, duplex, record, clip, cfg, seed):
            # Four QA are sampled before the first policy update; no queue crosses it.
            assert engine.global_steps == (0 if record['sample_index']<5 else 1)
            if record['sample_index']==1 and rank==2:
                from train.src.domain.trajectory import TrajectoryError
                raise TrajectoryError('rank_local_invalid_trajectory')
            t=trajectory();t['target']=float(seed%4)
            for c in t['chunks']:c['actions']=[]
            return t
        config=load_config(ROOT/'train/configs/minicpmo_full_grpo_judge.json')
        config.update(epochs=1,grpo_candidate_parallel=True,grpo_judge_pipeline_depth=2,
                      grpo_precheck_replay=True,grpo_activation_offload=False,save_every_steps=0)
        with ExitStack() as stack:
            prefix='train.src.training.grpo_judged.'
            stack.enter_context(patch(prefix+'Judge',FakeJudge))
            stack.enter_context(patch(prefix+'rollout',side_effect=fake_rollout))
            stack.enter_context(patch(prefix+'Clip',return_value=SimpleNamespace(horizon=6)))
            stack.enter_context(patch(prefix+'epoch_slots',return_value=list(range(6))))
            for name in ('torch.cuda.reset_peak_memory_stats','torch.cuda.max_memory_allocated','save',
                         'torch.distributed.destroy_process_group'):
                stack.enter_context(patch(prefix+name,return_value=0))
            run(engine,None,None,records,SimpleNamespace(output=Path(output),profile=False,max_steps=0),
                config,torch.device('cpu'),rank,world,{},MagicMock(),0.)
        report=json.loads(Path(output,f'memory_report.rank{rank}.json').read_text())
        assert report['optimizer_steps']==2 and report['valid_qa_this_run']==5
        assert report['excluded']==1
        rows=[json.loads(x) for x in Path(output,f'trajectories.rank{rank}.jsonl').read_text().splitlines()]
        assert len(rows)==5 and {r['candidate'] for r in rows}=={rank}
        Path(output,f'{rank}.passed').write_text('passed')
    finally:
        torch.distributed.destroy_process_group()


def test_four_rank_candidate_gradients_match_serial_group_mean(tmp_path):
    # Real collectives on four CPU ranks; no GPU models or occupied devices touched.
    mp.start_processes(distributed_worker,args=(4,'file://'+str(tmp_path/'rdzv'),str(tmp_path)),
                       nprocs=4,join=True,start_method='spawn')
    assert len(list(tmp_path.glob('*.passed')))==4
