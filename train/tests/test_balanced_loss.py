"""Balanced gate objective, gradients, missing classes and chunk invariance."""
import copy
import unittest
import torch
from train.src.models.sft_loss import grouped_loss
from train.src.domain.loss_metrics import statistics


class BalancedLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.head = torch.nn.Linear(4, 12)
        self.h = torch.randn(10, 4, requires_grad=True)
        self.y = torch.tensor([-100, 2, 2, 3, 6, 3, 5, 8, 4, 3])
        self.g = torch.tensor([0, 1, 1, 1, 2, 1, 1, 3, 1, 1])
        self.r = torch.tensor([0, 1, 1, 2, 0, 3, 3, 0, 3, 2])
        self.c = dict(decision_loss_mode='balanced_gate', decision_weight=1.2,
                      content_weight=.7, call_weight=.9, loss_projection_chunk=2,
                      gate_weight=.8, structure_weight=.2, silent_weight=.5, onset_weight=.5)

    def test_formula_gradients_chunking_and_contributions(self):
        head = copy.deepcopy(self.head)
        h = self.h.detach().clone().requires_grad_()
        ce = torch.nn.functional.cross_entropy(head(h[:-1]), self.y[1:], reduction='none')
        mean = lambda mask: ce[mask].mean()
        expected = 1.2 * (.4 * mean(self.r[1:] == 1) + .4 * mean(self.r[1:] == 2)
                          + .2 * mean(self.r[1:] == 3))
        expected = expected + .7 * mean(self.g[1:] == 2) + .9 * mean(self.g[1:] == 3)
        actual, raw = grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3, self.r)
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        torch.testing.assert_close(self.h.grad, h.grad)
        for a, b in zip(self.head.parameters(), head.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        stats = statistics(raw)
        self.assertEqual((stats['onset_tokens'], stats['continuation_speak_tokens'], stats['structure_tokens']), (2, 1, 3))
        self.assertAlmostEqual(stats['loss_gate_contribution'] + stats['loss_structure_contribution'], stats['loss_decision_contribution'], places=5)
        self.assertAlmostEqual(sum(stats['loss_' + k + '_contribution'] for k in ('decision', 'response', 'agent_instruction')), actual.item(), places=5)
        other, _ = grouped_loss(self.head, self.h, self.y, self.g, dict(self.c, loss_projection_chunk=100), 2, 3, self.r)
        torch.testing.assert_close(actual, other)

    def test_missing_and_zero_weight_classes_renormalize(self):
        for active_roles, extra, weights in [([2, 3], {}, {2:.8, 3:.2}),
                                             ([1], {}, {1:1.}),
                                             ([3], {}, {3:1.}),
                                             ([1, 2, 3], {'silent_weight':0}, {2:.8, 3:.2}),
                                             ([1, 2, 3], {'gate_weight':0}, {3:1.})]:
            with self.subTest(active_roles=active_roles, extra=extra):
                y = self.y.clone(); y[~torch.isin(self.r, torch.tensor(active_roles))] = -100
                c = dict(self.c, content_weight=0, call_weight=0, **extra)
                ce = torch.nn.functional.cross_entropy(self.head(self.h[:-1]), y[1:], reduction='none')
                expected = sum(w * ce[self.r[1:] == r].mean() for r, w in weights.items()) * c['decision_weight']
                actual, _ = grouped_loss(self.head, self.h, y, self.g, c, 2, 3, self.r)
                torch.testing.assert_close(actual, expected)

    def test_repeated_identical_silence_does_not_change_gate_balance(self):
        def loss(n):
            h = self.h[0].expand(n + 3, -1)
            y = torch.tensor([-100] + [2]*n + [3, 5])
            r = torch.tensor([0] + [1]*n + [2, 3])
            return grouped_loss(self.head, h, y, (r > 0).long(), self.c, 2, 3, r)[0]
        torch.testing.assert_close(loss(1), loss(100))

    def test_roles_required_and_validated(self):
        with self.assertRaisesRegex(ValueError, 'explicit decision_roles'):
            grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3)
        r = self.r.clone(); r[1] = 2
        with self.assertRaisesRegex(ValueError, 'disagree'):
            grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3, r)

class BalancedConfigTests(unittest.TestCase):
    def test_checked_in_modes_and_invalid_weights(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from train.src.infrastructure.config import load_config
        source = json.loads(Path('train/configs/minicpmo_full_sft_dp.json').read_text())
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for key, value in [('onset_weight', -1), ('gate_weight', float('nan')),
                               ('silent_weight', True), ('decision_loss_mode', 'bad')]:
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(source, **{key:value})))
                    with self.assertRaises(ValueError):
                        load_config(path)
            path.write_text(json.dumps(source))
            self.assertEqual(load_config(path)['decision_loss_mode'], 'balanced_gate')
            for key in ('decision_loss_mode', 'gate_weight', 'structure_weight', 'silent_weight', 'onset_weight'):
                source.pop(key)
            path.write_text(json.dumps(source))
            self.assertNotIn('decision_loss_mode', load_config(path))
