"""Fair two-by-two objectives: common scoring, independent advantages/reduction."""
import copy
import json
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.multiprocessing as mp

from train.src.domain.grpo import advantages, policy_loss
from train.src.infrastructure.config import ROOT, load_config
from train.src.models.local_policy import local_policy_loss
from train.src.training.grpo_judged import run, update_batch
from train.tests.test_local_grpo import CONFIG, LocalEngine, make_groups, sample, score, trajectory, with_credit

MODES = [(adv, agg) for adv in ('local', 'group_relative') for agg in ('token_mean', 'unit_weighted')]
TEMPLATES = {
    'baseline': ('group_relative', 'token_mean'),
    'local_advantage': ('local', 'token_mean'),
    'unit_aggregation': ('group_relative', 'unit_weighted'),
    'confusion_judge': ('local', 'unit_weighted'),
}


def reference(current, old, t, mode, aggregation):
    """Straightforward per-unit reference, without production indexing helpers."""
    credit = t['local_credit']
    groups = {}
    all_indices = []
    offset = 0
    for second, chunk in enumerate(t['chunks'], 1):
        if second >= credit.get('start_second', 1):
            for index, action in enumerate(chunk['actions'], offset):
                all_indices.append(index)
                if action['reply_id'] is not None:
                    key = ('reply', action['reply_id'])
                else:
                    key = ('idle' if chunk['idle_wait'] else 'control', second)
                groups.setdefault(key, []).append(index)
        offset += len(chunk['actions'])
    if aggregation == 'token_mean' and mode == 'group_relative':
        return policy_loss(current[all_indices], old[all_indices], t['advantage'], .2)[0]
    result = current.sum()*0
    local = [0.] * len(current)
    for unit in credit['units']:
        if unit['kind'] == 'reply':
            spans = [groups[('reply', unit['reply_id'])]]
        else:
            kind = 'control' if unit['kind'] == 'control' else 'idle'
            spans = [groups[(kind, second)] for second in unit['seconds']]
        adv = t['advantage'] if mode == 'group_relative' else unit['advantage']
        if aggregation == 'token_mean':
            for span in spans:
                for index in span:
                    local[index] += adv
        else:
            result += sum(policy_loss(current[span], old[span], adv, .2)[0]
                          for span in spans)/len(spans)*unit['weight']
    if aggregation == 'token_mean':
        result = policy_loss(current[all_indices], old[all_indices], current.new_tensor(local)[all_indices], .2)[0]
    return result


@pytest.mark.parametrize('mode,aggregation', MODES)
def test_objective_and_gradient_match_reference_with_neutral_and_prefix_tokens(mode, aggregation):
    qa = sample(((2, 4), (6, 7)))
    qa['question_time'] = 1
    t = trajectory([(3, 4, 'Two buckets.', 4), (5, 5, 'Again.', 2)])
    # A sampled, unowned control token is retained as a neutral structural unit.
    t['chunks'][7]['idle_wait'] = False
    reward = score(qa, t)
    t = dict(with_credit(t, reward), advantage=.7)
    size = sum(len(c['actions']) for c in t['chunks'])
    old = torch.full((size,), -.5, dtype=torch.float64)
    current = (old + torch.linspace(-.5, .5, size)).requires_grad_()
    other = current.detach().clone().requires_grad_()
    actual, _, _ = local_policy_loss(current, old, t, .2, advantage_mode=mode, aggregation=aggregation)
    expected = reference(other, old, t, mode, aggregation)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, current)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, other)[0])
    assert actual_grad[0] == 0  # Pre-question actions are excluded in every arm.
    if mode == 'group_relative':
        assert actual_grad[1] != 0  # Waiting in an eventually successful window.
    else:
        assert actual_grad[1] == 0


def test_rewards_and_complete_objective_are_unchanged_by_structural_zero_units():
    qa = sample(((2, 4), (6, 7)))
    t = trajectory([(3, 4, 'Two buckets.', 3)])
    t['chunks'][7]['idle_wait'] = False
    before = copy.deepcopy(t)
    results = [score(qa, t, dict(CONFIG, grpo_advantage_mode=mode, grpo_loss_aggregation=agg))
               for mode, agg in MODES]
    assert all(result == results[0] for result in results)
    assert t == before
    full = with_credit(t, results[0])
    nonzero = copy.deepcopy(full)
    nonzero['local_credit']['units'] = [u for u in nonzero['local_credit']['units'] if u['advantage']]
    size = sum(len(c['actions']) for c in t['chunks'])
    old = torch.full((size,), -.5, dtype=torch.float64)
    current = (old + .07).requires_grad_()
    a = local_policy_loss(current, old, full, .2)[0]
    b = local_policy_loss(current, old, nonzero, .2)[0]
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a, current, retain_graph=True)[0], torch.autograd.grad(b, current)[0])


def test_token_local_sums_overlapping_fn_and_does_not_apply_unit_weights():
    qa = sample(((2, 4), (3, 5)))
    t = trajectory(horizon=8)
    r = score(qa, t, dict(CONFIG, reward_weights=dict(CONFIG['reward_weights'], tn=0)))
    local = with_credit(t, r)
    current = torch.full((8,), -.5, requires_grad=True)
    old = current.detach().clone()
    value, _, _ = local_policy_loss(current, old, local, .2, aggregation='token_mean')
    value.backward()
    assert current.grad[0] == pytest.approx(-1/8)  # Waiting advantage +1 even when its aggregation weight is zero.
    assert current.grad[2] == pytest.approx(2/8)   # Two overlapping missed windows.
    assert value.item() == pytest.approx((3+3-4)/8)


def test_templates_differ_only_in_two_selectors_and_defaults_validate(tmp_path):
    raw = [json.loads((ROOT/f'train/configs/minicpmo_full_grpo_{name}_dp4.json').read_text()) for name in TEMPLATES]
    selectors = {'grpo_advantage_mode', 'grpo_loss_aggregation'}
    common = [{k:v for k,v in c.items() if k not in selectors} for c in raw]
    assert all(c == common[0] for c in common)
    for name, pair in TEMPLATES.items():
        c = load_config(ROOT/f'train/configs/minicpmo_full_grpo_{name}_dp4.json')
        assert (c['grpo_advantage_mode'], c['grpo_loss_aggregation']) == pair
    path = tmp_path/'config.json'
    path.write_text(json.dumps(common[0]))
    default = load_config(path)
    assert (default['grpo_advantage_mode'], default['grpo_loss_aggregation']) == ('local','unit_weighted')
    for change in ({'grpo_advantage_mode':'bad'}, {'grpo_loss_aggregation':'bad'}, {'reward_mode':'confusion_judge'}):
        path.write_text(json.dumps(dict(raw[0], **change)))
        with pytest.raises(ValueError):
            load_config(path)


class AblationEngine(LocalEngine):
    def __init__(self, mode, aggregation):
        super().__init__()
        self.mode, self.aggregation = mode, aggregation

    def __call__(self, record, clip, trajectory):
        size = sum(len(c['actions']) for c in trajectory['chunks'])
        old = torch.full((size,), -.5)
        current = old + self.parameter*record['x']
        if 'local_credit' not in trajectory:
            return current.detach(), (current.detach()-old).abs().max()
        value, ratio, info = local_policy_loss(current, old, trajectory, .2,
            advantage_mode=self.mode, aggregation=self.aggregation)
        return value, dict(info, replay_max_logprob_error=0., ratio_mean=ratio.mean().item())


def batch(count):
    result = make_groups(count)
    for group in result:
        rewards = [score(sample(), t) for t in group['trajectories']]
        adv, _ = advantages([r['total'] for r in rewards])
        group['advantages'] = adv.tolist()
    return result


def expected_batch(engine, groups):
    return sum(sum(engine(g['record'], None, dict(t, local_credit=c, advantage=a))[0]
        for t,c,a in zip(g['trajectories'],g['local_credits'],g['advantages']))/4 for g in groups)/len(groups)


@pytest.mark.parametrize('mode,aggregation', MODES)
def test_accumulation_and_tail_batch_equal_serial_reference(mode, aggregation):
    for count in (1,3,4):
        engine = AblationEngine(mode, aggregation)
        expected = AblationEngine(mode, aggregation)
        groups = batch(count)
        reference_loss = expected_batch(expected, groups)
        grad = torch.autograd.grad(reference_loss, expected.parameter)[0]
        config = dict(CONFIG, grpo_advantage_mode=mode, grpo_loss_aggregation=aggregation,
            gradient_accumulation_steps=4, replay_logprob_tolerance=.15, grpo_activation_offload=False)
        with patch('train.src.training.grpo_judged.all_true', side_effect=lambda ok, device:ok):
            result = update_batch(engine, groups, config, torch.device('cpu'))
        torch.testing.assert_close(engine.parameter, expected.parameter.detach()-.01*grad)
        assert result['loss'] == pytest.approx(reference_loss.item())
        assert engine.global_steps == 1 and result['padding_slots'] == 4-count


def distributed_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', rank=rank, world_size=world,
        init_method=rendezvous, timeout=timedelta(seconds=60))
    class DistributedEngine(AblationEngine):
        def step(self):
            if (self.micro_steps+1)%self.gas == 0:
                torch.distributed.all_reduce(self.parameter.grad)
                self.parameter.grad.div_(world)
            super().step()
    try:
        for mode, aggregation in MODES:
            for count in (1,3,4):
                groups = batch(count)
                expected = AblationEngine(mode, aggregation)
                reference_loss = expected_batch(expected, groups)
                grad = torch.autograd.grad(reference_loss, expected.parameter)[0]
                local = [dict(g, trajectories=[g['trajectories'][rank]],
                    local_credits=[g['local_credits'][rank]], advantages=[g['advantages'][rank]]) for g in groups]
                engine = DistributedEngine(mode, aggregation)
                config = dict(CONFIG, grpo_advantage_mode=mode, grpo_loss_aggregation=aggregation,
                    gradient_accumulation_steps=4, replay_logprob_tolerance=.15,
                    grpo_activation_offload=False, grpo_candidate_parallel=True)
                result = update_batch(engine, local, config, torch.device('cpu'))
                torch.testing.assert_close(engine.parameter, expected.parameter.detach()-.01*grad)
                assert result['loss'] == pytest.approx(reference_loss.item())
                assert result['candidate_trajectories'] == count*world
    finally:
        torch.distributed.destroy_process_group()


def test_four_cpu_candidate_ranks_match_all_four_objectives(tmp_path):
    mp.start_processes(distributed_worker, args=(4,'file://'+str(tmp_path/'rdzv')),
                       nprocs=4, join=True, start_method='spawn')


@pytest.mark.parametrize('mode,aggregation', MODES)
def test_trainer_fixed_sampling_budget_and_skip_semantics(tmp_path, mode, aggregation):
    class Judge:
        def __init__(self,*args):pass
        def ready(self):pass
        def snapshot(self):return dict(requests=1,cache_hits=0,failures=1,retries=0)
        def score_group(self,qa,ts):
            if qa['question']=='fail':raise TimeoutError()
            return [{} for _ in ts]
    config = load_config(ROOT/'train/configs/minicpmo_full_grpo_confusion_judge.json')
    config.update(epochs=1, save_every_steps=0, grpo_activation_offload=False,
        grpo_judge_pipeline_depth=2, grpo_advantage_mode=mode, grpo_loss_aggregation=aggregation)
    records = [dict(sample_index=i, sample=dict(sample(),question='fail' if i==0 else 'wait'),x=.1) for i in range(3)]
    engine = AblationEngine(mode, aggregation)
    prefix = 'train.src.training.grpo_judged.'
    with ExitStack() as stack:
        stack.enter_context(patch(prefix+'Judge', Judge))
        stack.enter_context(patch(prefix+'rollout', side_effect=lambda *args:trajectory()))
        stack.enter_context(patch(prefix+'Clip', return_value=SimpleNamespace(horizon=10)))
        stack.enter_context(patch(prefix+'epoch_slots', return_value=[0,1,2]))
        stack.enter_context(patch(prefix+'broadcast', side_effect=lambda packet,rank:packet))
        stack.enter_context(patch(prefix+'all_true', side_effect=lambda ok,device:ok))
        for name in ('torch.cuda.reset_peak_memory_stats','torch.cuda.max_memory_allocated',
                     'torch.distributed.barrier','torch.distributed.destroy_process_group','save'):
            stack.enter_context(patch(prefix+name, return_value=0))
        run(engine,None,None,records,SimpleNamespace(output=tmp_path,profile=False,max_steps=0),
            config,torch.device('cpu'),0,4,{},MagicMock(),0.)
    report = json.loads((tmp_path/'memory_report.rank0.json').read_text())
    assert report['sampled_qa_groups'] == 3 and report['rollout_attempts'] == report['completed_rollouts'] == 12
    assert report['judge_failed_groups'] == 1 and report['scored_qa_groups'] == 2
    assert report['grpo_advantage_mode'] == mode and report['grpo_loss_aggregation'] == aggregation
    if mode == 'group_relative':
        assert engine.global_steps == 0 and report['zero_variance_groups'] == 2
        assert report['update_skip_rate'] == 1
    else:
        assert engine.global_steps == 1 and report['zero_variance_groups'] == 0
    saved = [json.loads(line) for line in (tmp_path/'trajectories.rank0.jsonl').read_text().splitlines()]
    assert all(row['optimization']['advantage_mode'] == mode for row in saved)
    assert all(row['optimization']['loss_aggregation'] == aggregation for row in saved)


@pytest.mark.parametrize('mode,aggregation', MODES)
def test_normal_graph_entrypoint_uses_both_selectors(mode, aggregation):
    from train.tests.test_grpo import ReplayGradientTests
    with patch('train.src.models.streaming.user_request', return_value='question'), \
         patch('train.src.models.streaming.system_prompt', return_value='system'):
        graph, record, clip = ReplayGradientTests().setup_graph()
        graph.eval()
        graph.config.update(grpo_advantage_mode=mode, grpo_loss_aggregation=aggregation)
        actions = [dict(offset=i, turn_open=0<i<5, reply_id=0 if i<5 else None, old_log_prob=0.)
                   for i in range(6)]
        t = dict(chunks=[dict(ids=[301,97,300,98,303,304,305], actions=actions, idle_wait=False)])
        with torch.no_grad():
            old, _ = graph(record, clip, t)
        for action, value in zip(actions, old.tolist()):
            action['old_log_prob'] = value
        t['advantage'] = .7
        t['local_credit'] = dict(version=1, units=[
            dict(kind='reply', reply_id=0, action_count=5, advantage=1., weight=1.,
                 response_index=0, base_advantage=1., blocked_fn_advantage=0.),
            dict(kind='control', seconds=[1], advantage=0., weight=1.)])
        value, info = graph(record, clip, t)
        expected = {('local','token_mean'):-5/6, ('local','unit_weighted'):-1.,
                    ('group_relative','token_mean'):-.7, ('group_relative','unit_weighted'):-1.4}
        assert value.item() == pytest.approx(expected[mode,aggregation], abs=1e-5)
        assert info['replay_max_logprob_error'] < 1e-5
        value.backward()
        assert graph.model.llm.lm_head.weight.grad.abs().sum() > 0


def test_checkpoint_identities_distinguish_all_four_experiments(tmp_path):
    from train.src.infrastructure.checkpoints import fingerprints
    for name in ('train.json','validation.json','manifest.json'):
        (tmp_path/name).write_text('{}')
    base = load_config(ROOT/'train/configs/minicpmo_full_grpo_confusion_judge_dp4.json')
    with patch('benchmark.data.agent_token.base_identity', return_value={}), \
         patch('train.src.infrastructure.tokens.load_training_tokenizer', return_value=None), \
         patch('train.src.infrastructure.checkpoints.tokenizer_metadata', return_value={}), \
         patch('train.src.infrastructure.judge.identity', return_value={}):
        identities = [fingerprints(tmp_path, dict(base,grpo_advantage_mode=mode,grpo_loss_aggregation=agg))
                      for mode,agg in MODES]
    assert len({json.dumps(i, sort_keys=True) for i in identities}) == 4
