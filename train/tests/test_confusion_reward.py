"""Trajectory reward arithmetic and anti-spam/always-silent failure modes."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from train.tests.test_grpo_judge import sample, trajectory
from train.src.domain.rewards import judged_reward
from train.src.infrastructure.config import load_config
from benchmark.data.task_text import event


class ConfusionRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config('train/configs/minicpmo_full_grpo_confusion_judge.json')
        self.qa = sample()
        self.qa['responses'].append(dict(response='Green now.', window_start='00:05', window_end='00:05'))

    def roll(self, times, n=8):
        return dict(events=[event('A correct answer.', t, complete=True) for t in times],
                    chunks=[dict(new_response=t in times, idle_wait=t not in times) for t in range(1,n+1)])

    def answers(self, score=2):
        return {i:dict(status='complete', response_score=score, call_score=None) for i in (0,1)}

    def reward(self, times, score=2, **kwargs):
        return judged_reward(self.qa,self.roll(times),self.answers(score),dict(self.config, **kwargs))

    def test_hand_examples_without_tn(self):
        config = dict(self.config, reward_weights=dict(self.config['reward_weights'], tn=0))
        for times,score,expected in [([2,5],2,1),([2],2,0),([],2,-1),
                                     ([2,5,1,4,6,7],2,0),([2,5],0,-1)]:
            with self.subTest(times=times,score=score):
                r=judged_reward(self.qa,self.roll(times),self.answers(score),config)
                self.assertAlmostEqual(r['total'],expected)
                self.assertEqual(r['tp']+r['fn'],2)
                self.assertAlmostEqual(r['total'],sum(r[k+'_contribution'] for k in ('tp','fp','fn','tn','invalid_call')))
        # Repeated parsed events in the same window still each incur FP.
        t=self.roll([2,5]);t['events'] += [event('Again.',2,complete=True) for _ in range(100)]
        r=judged_reward(self.qa,t,self.answers(),config)
        self.assertEqual(r['fp'],100)
        self.assertAlmostEqual(r['total'],-24)

    def test_content_score_levels_and_always_silent(self):
        rewards=[self.reward([2,5],v)['total'] for v in (0,.5,1,1.5,2)]
        self.assertEqual(rewards,sorted(rewards))
        self.assertAlmostEqual(self.reward([])['total'],-.9)
        self.assertAlmostEqual(self.reward([2,5])['total'],1.1)
        self.assertLess(self.reward([2,5],0)['total'],0)

    def test_tn_is_duration_normalized_and_continuation_is_not_waiting(self):
        for n in (8,800):
            r=judged_reward(self.qa,self.roll([],n),{},self.config)
            self.assertAlmostEqual(r['tn_contribution'],.1)
        t=self.roll([]); t['chunks'][3]['idle_wait']=False  # continuing a response, no fresh onset
        r=judged_reward(self.qa,t,{},self.config)
        self.assertEqual(r['negative_seconds'],5)
        self.assertEqual(r['tn'],4)
        self.assertAlmostEqual(r['tn_contribution'],.06)
        self.qa['question_time']='00:01'
        r=judged_reward(self.qa,self.roll([]),{},self.config)
        self.assertEqual(r['negative_seconds'],4)  # second 1 is pre-question

    def test_empty_attempt_and_missing_telemetry(self):
        t=self.roll([]);t['chunks'][3].update(new_response=True,idle_wait=False)
        r=judged_reward(self.qa,t,{},self.config)
        self.assertEqual((r['fp'],r['unparsed_response_starts']),(1,1))
        self.assertFalse(r['silent'])
        self.assertLess(r['total'],self.reward([])['total'])
        del t['chunks'][3]['idle_wait']
        with self.assertRaisesRegex(ValueError,'telemetry'):
            judged_reward(self.qa,t,{},self.config)

    def test_agent_invalid_call_never_earns_instruction_credit(self):
        qa=sample(True);answers={0:dict(status='complete',response_score=2,call_score=2)}
        good=trajectory('Checking.<|task_start|>Write a red-light note.')
        for t in (good,):
            for c in t['chunks']:c['idle_wait']=not c['new_response']
        r=judged_reward(qa,good,answers,self.config)
        self.assertAlmostEqual(r['total'],1.1)
        bad=copy.deepcopy(good);bad['events']=[event('Checking.<|task_start|>',2,complete=True)]
        r=judged_reward(qa,bad,answers,self.config)
        self.assertEqual(r['call'],0)
        self.assertEqual(r['invalid_calls'],1)
        self.assertAlmostEqual(r['total'],0)
        absent=copy.deepcopy(bad);absent['events']=[event('Checking.',2,complete=True)]
        self.assertEqual(judged_reward(qa,absent,answers,self.config)['call'],0)

    def test_failed_judge_and_bad_scores_exclude_not_zero(self):
        for value in (None,dict(status='failed'),dict(status='complete',response_score=3),
                      dict(status='complete',response_score=True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                judged_reward(self.qa,self.roll([2]),{0:value},self.config)
        with self.assertRaisesRegex(ValueError,'coverage'):
            judged_reward(self.qa,self.roll([],n=4),{},self.config)

    def test_config_is_explicit_and_old_mode_remains(self):
        self.assertNotIn('reward_mode',load_config('train/configs/minicpmo_full_grpo_judge.json'))
        with TemporaryDirectory() as directory:
            p=Path(directory)/'config.json'
            for update in ({'grpo_horizon':'prefix'},{'reward_backend':'rules'},
                           {'reward_mode':'bad'},{'reward_weights':dict(self.config['reward_weights'],fp=-1)}):
                p.write_text(json.dumps(dict(self.config,**update)))
                with self.assertRaises(ValueError):load_config(p)

if __name__=='__main__':unittest.main()
