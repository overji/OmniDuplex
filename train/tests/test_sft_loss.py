"""Telemetry must leave the previous SFT value and gradients unchanged."""
import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from train.src.models.sft_loss import grouped_loss
from train.src.domain.loss_metrics import statistics


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.head = torch.nn.Linear(4, 12)
        self.hidden = torch.randn(8, 4, requires_grad=True)
        self.labels = torch.tensor([-100, 2, 3, 6, 7, 5, 8, 4])
        self.groups = torch.tensor([0, 1, 1, 2, 2, 1, 3, 1])
        self.config = dict(decision_weight=1.2, content_weight=.7, call_weight=.9,
                           loss_projection_chunk=2)

    def test_value_and_gradients_match_previous_objective_exactly(self):
        reference_head = copy.deepcopy(self.head)
        reference_hidden = self.hidden.detach().clone().requires_grad_()
        reference = reference_hidden.sum() * 0
        for group, key in ((1, 'decision_weight'), (2, 'content_weight'), (3, 'call_weight')):
            positions = torch.where((self.labels[1:] != -100) & (self.groups[1:] == group))[0]
            for ix in positions.split(self.config['loss_projection_chunk']):
                def project(hidden, target):
                    return torch.nn.functional.cross_entropy(reference_head(hidden).float(), target, reduction='sum')
                reference = reference + self.config[key] * checkpoint(
                    project, reference_hidden[ix], self.labels[ix+1], use_reentrant=False) / len(positions)
        loss, raw = grouped_loss(self.head, self.hidden, self.labels, self.groups, self.config, 2, 3)
        self.assertEqual(loss.item(), reference.item())
        saved = list(raw)
        loss.backward(); reference.backward()
        self.assertEqual(raw, saved)  # checkpoint recomputation has no metric side effects
        torch.testing.assert_close(self.hidden.grad, reference_hidden.grad, rtol=0, atol=0)
        for actual, expected in zip(self.head.parameters(), reference_head.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        stats = statistics(raw)
        self.assertEqual([stats[n + '_tokens'] for n in ('silence', 'speak', 'control',
            'response', 'agent_instruction', 'decision')], [1, 1, 2, 2, 1, 4])
        contribution = sum(stats['loss_' + name + '_contribution'] for name in
                           ('decision', 'response', 'agent_instruction'))
        self.assertAlmostEqual(contribution, loss.item(), places=5)
        self.assertAlmostEqual(raw[5], sum(raw[:3]), places=5)

    def test_missing_or_disabled_class_is_null_not_a_zero_loss(self):
        labels = self.labels.clone(); labels[6] = -100
        config = {**self.config, 'content_weight': 0.}
        _, raw = grouped_loss(self.head, self.hidden, labels, self.groups, config, 2, 3)
        stats = statistics(raw)
        for name in ('agent_instruction', 'response'):
            self.assertIsNone(stats['loss_' + name])
            self.assertEqual(stats[name + '_tokens'], 0)
            self.assertEqual(stats[name + '_qa'], 0)

    def test_pooled_token_means_and_objective_contributions_are_distinct(self):
        raw = [0.] * 15
        raw[0], raw[6], raw[12] = 14., 4., 1.
        stats = statistics(raw, qa_count=2, presence=[2, 0, 0, 0, 0, 0])
        self.assertEqual(stats['loss_silence'], 3.5)
        self.assertEqual(stats['silence_qa'], 2)
        self.assertEqual(stats['loss_decision_contribution'], .5)

    def test_no_active_targets_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No supervised'):
            grouped_loss(self.head, self.hidden, self.labels, self.groups,
                         {**self.config, 'decision_weight': 0., 'content_weight': 0., 'call_weight': 0.}, 2, 3)

    def test_plain_ce_matches_joint_token_mean_and_gradients(self):
        reference_head = copy.deepcopy(self.head)
        reference_hidden = self.hidden.detach().clone().requires_grad_()
        labels = self.labels.clone()
        labels[4] = -100  # masked targets must not contribute to the denominator
        reference = torch.nn.functional.cross_entropy(reference_head(reference_hidden[:-1]), labels[1:])
        config = dict(self.config, decision_loss_mode='plain_ce',
                      decision_weight=1., content_weight=1., call_weight=1.)
        loss, raw = grouped_loss(self.head, self.hidden, labels, self.groups, config, 2, 3)
        torch.testing.assert_close(loss, reference)
        loss.backward(); reference.backward()
        torch.testing.assert_close(self.hidden.grad, reference_hidden.grad)
        for actual, expected in zip(self.head.parameters(), reference_head.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        self.assertAlmostEqual(sum(raw[12:15]), loss.item(), places=6)

    def test_plain_ce_fine_onset_metrics_do_not_change_loss_or_gradient(self):
        labels = self.labels.clone(); labels[-1] = 3
        roles = torch.tensor([0, 1, 3, 0, 0, 6, 0, 4])
        cfg = dict(self.config, decision_loss_mode='plain_ce', decision_weight=1., content_weight=1., call_weight=1.)
        h = self.hidden.detach().clone().requires_grad_()
        head = copy.deepcopy(self.head)
        original, _ = grouped_loss(head, h, labels, self.groups, cfg, 2, 3)
        actual, raw = grouped_loss(self.head, self.hidden, labels, self.groups, cfg, 2, 3, event_roles=roles)
        torch.testing.assert_close(actual, original)
        actual.backward(); original.backward()
        torch.testing.assert_close(self.hidden.grad, h.grad)
        for a, b in zip(self.head.parameters(), head.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        stats = statistics(raw)
        assert stats['first_onset_tokens'] == stats['later_onset_tokens'] == 1
        assert stats['loss_first_onset'] is not None and stats['loss_later_onset'] is not None


if __name__ == '__main__':
    unittest.main()
