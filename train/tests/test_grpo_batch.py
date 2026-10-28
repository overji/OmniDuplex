"""Compare actual QA/candidate averaging with a differentiable CPU reference."""
import unittest
from unittest.mock import patch, MagicMock
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from train.src.training.grpo_judged import update_batch


class Engine:
    def __init__(self, gas=4):
        self.parameter = torch.nn.Parameter(torch.tensor(.7))
        self.gas, self.micro_steps, self.global_steps = gas, 0, 0
        self.parameter_history = []

    def __call__(self, record, clip, trajectory):
        if 'advantage' not in trajectory:
            return torch.tensor(0.), torch.tensor(0.)
        loss = (self.parameter * record['x'] - trajectory['target']) ** 2 * trajectory['advantage']
        return loss, dict(replay_max_logprob_error=0., ratio_mean=1.)

    def backward(self, loss):
        (loss / self.gas).backward()

    def step(self):
        self.micro_steps += 1
        if self.micro_steps % self.gas == 0:
            with torch.no_grad():self.parameter.add_(self.parameter.grad, alpha=-.01)
            self.parameter.grad = None
            self.global_steps += 1
        self.parameter_history.append(self.parameter.item())

    def eval(self):
        return self


def groups(count):
    return [dict(record={'x': i+1}, clip=None,
                 trajectories=[dict(target=float(j+i), chunks=[dict(actions=[])]) for j in range(4)],
                 advantages=[1., -.5, .5, -1.]) for i in range(count)]


class AccumulationTests(unittest.TestCase):
    @patch('train.src.training.grpo_judged.all_true', side_effect=lambda okay, device: okay)
    def test_full_and_partial_batch_equal_reference_and_update_only_once(self, _):
        config = dict(gradient_accumulation_steps=4, replay_logprob_tolerance=.15, grpo_activation_offload=False)
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                engine = Engine()
                parameter = engine.parameter.detach().clone().requires_grad_()
                batch = groups(count)
                expected_loss = sum(sum((parameter*g['record']['x']-t['target'])**2*a
                                       for t, a in zip(g['trajectories'], g['advantages']))/4 for g in batch)/count
                expected_loss.backward()
                expected = parameter.detach() - .01 * parameter.grad
                stats = update_batch(engine, batch, config, torch.device('cpu'))
                torch.testing.assert_close(engine.parameter, expected)
                self.assertEqual(engine.global_steps, 1)
                self.assertEqual(engine.micro_steps, 4)
                self.assertEqual(engine.parameter_history[:3], [torch.tensor(.7).item()] * 3)
                self.assertEqual(stats['candidate_trajectories'], count*4)
                self.assertEqual(stats['global_batch_size'], count)
                self.assertEqual(stats['padding_slots'], 4-count)
                self.assertAlmostEqual(stats['loss'], expected_loss.item(), places=5)

    @patch('train.src.training.grpo_judged.all_true', return_value=False)
    def test_replay_failure_stops_before_update(self, _):
        engine = Engine()
        with self.assertRaisesRegex(RuntimeError, 'replay mismatch'):
            update_batch(engine, groups(4), dict(gradient_accumulation_steps=4, replay_logprob_tolerance=.15), torch.device('cpu'))
        self.assertEqual(engine.global_steps, 0)

    def test_trainer_skips_failed_and_zero_groups_and_flushes_partial_epoch(self):
        from train.src.infrastructure.config import ROOT, load_config
        from train.src.training.grpo_judged import run
        from train.tests.test_grpo_judge import sample, trajectory
        class FakeJudge:
            def __init__(self, settings):pass
            def ready(self):pass
            def snapshot(self):return dict(requests=10, cache_hits=0, failures=1, retries=0)
            def score_group(self, qa, trajectories):
                if qa['question'] == 'fail':raise TimeoutError()
                scores = [2] * 4 if qa['question'] == 'zero' else [0, .5, 1.5, 2]
                return [{0:dict(status='complete',response_score=s,call_score=None)} for s in scores]
        for count in (5, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                config = load_config(ROOT / 'train/configs/minicpmo_full_grpo_judge.json')
                config.update(epochs=1, grpo_activation_offload=False, save_every_steps=0)
                records=[]
                for i in range(count):
                    qa=sample();qa['question']='fail' if i==0 else 'zero' if i==1 else str(i)
                    records.append(dict(sample_index=i, sample=qa, x=i+1))
                def fake_rollout(model, duplex, record, clip, cfg, seed):
                    t=trajectory();t['target']=float(seed%4)
                    for chunk in t['chunks']:chunk['actions']=[]
                    return t
                target='train.src.training.grpo_judged.'
                stack.enter_context(patch(target+'Judge',FakeJudge))
                stack.enter_context(patch(target+'rollout',side_effect=fake_rollout))
                stack.enter_context(patch(target+'Clip',return_value=SimpleNamespace(horizon=6)))
                stack.enter_context(patch(target+'epoch_slots',return_value=list(range(count))))
                stack.enter_context(patch(target+'broadcast',side_effect=lambda packet,rank:packet))
                stack.enter_context(patch(target+'all_true',side_effect=lambda okay,device:okay))
                for name in ('torch.cuda.reset_peak_memory_stats','torch.cuda.max_memory_allocated',
                             'torch.distributed.barrier','torch.distributed.destroy_process_group','save'):
                    stack.enter_context(patch(target+name,return_value=0))
                engine=Engine();output=Path(directory)
                run(engine,None,None,records,SimpleNamespace(output=output,profile=False,max_steps=0),
                    config,torch.device('cpu'),0,4,{},MagicMock(),0.)
                report=json.loads((output/'memory_report.rank0.json').read_text())
                self.assertEqual(report['valid_qa_this_run'],count-2)
                self.assertEqual(report['excluded'],1)
                self.assertEqual(report['zero_variance_groups'],1)
                self.assertEqual(engine.global_steps,1)
                self.assertEqual(engine.micro_steps,4)


if __name__ == '__main__':
    unittest.main()
