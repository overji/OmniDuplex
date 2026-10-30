"""Event objective arithmetic, gradients, metadata, and backward compatibility."""
import copy
import unittest
import torch
from train.src.models.sft_loss import grouped_loss
from train.src.models.event_loss import event_weights
from train.src.domain.loss_metrics import statistics
from train.src.domain.trajectory import supervision_metadata


class EventLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.head = torch.nn.Linear(4, 12)
        # before, first onset, content, continuation, content, marker, instruction,
        # end, after, later onset, content, marker, instruction, end, after
        self.y = torch.tensor([-100, 2, 3, 8, 3, 9, 7, 8, 5, 2, 3, 9, 7, 9, 5, 2])
        self.g = torch.tensor([0, 1, 1, 2, 1, 2, 1, 3, 1, 1, 1, 2, 1, 3, 1, 1])
        self.r = torch.tensor([0, 1, 3, 0, 5, 0, 7, 0, 6, 2, 4, 0, 7, 0, 6, 2])
        self.e = torch.tensor([-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1, -1])
        self.coarse = torch.where(self.r <= 2, 1, torch.where(self.r <= 4, 2, 3)) * (self.r > 0)
        self.h = torch.randn(len(self.y), 4, requires_grad=True)
        self.c = dict(decision_loss_mode='event_balanced', decision_weight=1., content_weight=1.,
                      call_weight=1., loss_projection_chunk=2)

    def run_loss(self, **kwargs):
        return grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3,
                            self.coarse, **dict(event_roles=self.r, event_ids=self.e, **kwargs))

    def test_hand_formula_gradients_chunking_and_metrics(self):
        head = copy.deepcopy(self.head); h = self.h.detach().clone().requires_grad_()
        ce = torch.nn.functional.cross_entropy(head(h[:-1]), self.y[1:], reduction='none')
        roles, groups, events = self.r[1:], self.g[1:], self.e[1:]
        mean = lambda mask: ce[mask].mean()
        event_mean = lambda mask: torch.stack([mean(mask & (events == e)) for e in events[mask].unique()]).mean()
        expected = .4 * mean((roles == 1) | (roles == 2))
        expected += .2 * sum(mean(roles == r) for r in (3, 4))
        expected += .2 / 3 * sum(event_mean(roles == r) for r in (5, 6, 7))
        expected += event_mean(groups == 2) + event_mean(groups == 3)
        actual, raw = self.run_loss()
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        torch.testing.assert_close(self.h.grad, h.grad)
        for p, q in zip(self.head.parameters(), head.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        stats = statistics(raw)
        self.assertEqual(stats['later_onset_tokens'], 1)
        self.assertEqual(stats['silence_after_tokens'], 2)
        self.assertAlmostEqual(sum(stats['loss_' + k + '_contribution'] for k in
            ('decision', 'response', 'agent_instruction')), actual.item(), places=5)
        self.assertAlmostEqual(sum(stats['loss_' + k + '_contribution'] for k in
            ('silence_before', 'silence_after', 'first_onset', 'later_onset', 'continue', 'end', 'marker')),
            stats['loss_decision_contribution'], places=5)
        saved = list(raw)
        self.c['loss_projection_chunk'] = 100
        other, _ = self.run_loss()
        torch.testing.assert_close(actual, other)
        self.assertEqual(raw, saved)

    def test_missing_branches_and_event_budget(self):
        weights = event_weights(self.y[1:], self.g[1:], self.r[1:], self.e[1:], self.c, 2, 3)
        for event in (0, 1):
            self.assertAlmostEqual(float(weights[(self.g[1:] == 2) & (self.e[1:] == event)].sum()), .5)
        # Single response with no silence, continuation, marker, or instruction.
        w = event_weights(torch.tensor([3, 8, 5]), torch.tensor([1, 2, 1]),
                          torch.tensor([3, 0, 6]), torch.tensor([0, 0, 0]), self.c, 2, 3)
        torch.testing.assert_close(w, torch.tensor([.8, 1., .2]))
        w = event_weights(torch.tensor([2]), torch.tensor([1]), torch.tensor([2]),
                          torch.tensor([-1]), self.c, 2, 3)
        torch.testing.assert_close(w, torch.ones(1))

    def test_silence_partition_does_not_change_loss_or_gradients(self):
        idle = (self.r == 1) | (self.r == 2)
        reference = None
        # Mixed, all-before, all-after, and swapped assignments cover absent groups.
        for assignment in ([1, 2, 2], [1, 1, 1], [2, 2, 2], [2, 1, 1]):
            roles = self.r.clone()
            roles[idle] = torch.tensor(assignment)
            head = copy.deepcopy(self.head)
            hidden = self.h.detach().clone().requires_grad_()
            loss, raw = grouped_loss(head, hidden, self.y, self.g, self.c, 2, 3,
                self.coarse, event_roles=roles, event_ids=self.e)
            loss.backward()
            values = [loss.detach(), hidden.grad] + [p.grad for p in head.parameters()]
            if reference is None:
                reference = [v.clone() for v in values]
            else:
                for actual, expected in zip(values, reference):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            weights = event_weights(self.y[1:], self.g[1:], roles[1:], self.e[1:], self.c, 2, 3)
            torch.testing.assert_close(weights[idle[1:]], torch.full((3,), .4 / 3))
            ce = torch.nn.functional.cross_entropy(head(hidden[:-1]), self.y[1:], reduction='none')
            stats = statistics(raw)
            self.assertAlmostEqual(stats['loss_silence_before_contribution'] +
                stats['loss_silence_after_contribution'], float(.4 * ce[idle[1:]].mean().detach()), places=6)

    def test_wrong_onset_probability_increases_silence_loss(self):
        head = torch.nn.Identity()
        losses = []
        for speak_logit in (0., 4.):
            h = torch.tensor([[0., speak_logit], [0., 0.]], requires_grad=True)
            loss, _ = grouped_loss(head, h, torch.tensor([-100, 0]), torch.tensor([0, 1]),
                self.c, 0, 1, event_roles=torch.tensor([0, 2]), event_ids=torch.tensor([-1, -1]))
            loss.backward()
            self.assertGreater(h.grad[0, 1].item(), 0)
            losses.append(loss.item())
        self.assertGreater(losses[1], losses[0])

    def test_duplicate_content_preserves_per_event_budget(self):
        def weights(n):
            return event_weights(torch.tensor([8]*n + [9]), torch.tensor([2]*(n+1)),
                torch.zeros(n+1, dtype=torch.long), torch.tensor([0]*n + [1]), self.c, 2, 3)
        for n in (1, 100):
            w = weights(n)
            torch.testing.assert_close(w[:n].sum(), torch.tensor(.5))
            self.assertAlmostEqual(w[-1].item(), .5)

    def test_full_video_config_validation(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from train.src.infrastructure.config import load_config
        source = Path('train/configs/minicpmo_full_sft_full_video_dp.json')
        config = load_config(source)
        self.assertEqual(config['max_sequence_tokens'], 65536)
        self.assertEqual(config['sft_horizon'], 'full_video')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for change in ({'supervise_silence': False}, {'post_response_silence_seconds': 10},
                           {'sft_horizon': 'bad'}):
                path.write_text(json.dumps(dict(json.loads(source.read_text()), **change)))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_invalid_metadata_and_config(self):
        with self.assertRaisesRegex(ValueError, 'aligned'):
            grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3)
        e = self.e.clone(); e[3] = -1
        with self.assertRaisesRegex(ValueError, 'event ID'):
            grouped_loss(self.head, self.h, self.y, self.g, self.c, 2, 3,
                         event_roles=self.r, event_ids=e)
        from train.src.infrastructure.config import load_config
        config = load_config('train/configs/minicpmo_full_sft_event_balanced_dp.json')
        self.assertEqual(config['decision_loss_mode'], 'event_balanced')
        self.assertEqual(load_config('train/configs/minicpmo_full_sft_dp.json')['decision_loss_mode'], 'balanced_gate')


class EventMetadataTests(unittest.TestCase):
    def test_multichunk_agent_marker_and_idle(self):
        from train.src.domain.targets import target_chunks
        from train.tests.helpers import ByteTokenizer
        tokenizer = ByteTokenizer()
        # Group 1 ordinary separator bytes must remain marker roles across chunks.
        record = {'sample': {'mode': 'agent'}, 'reference': {'response': 'ok',
            'tool_calls': [{'name': 'task_start', 'arguments': {'instruction': 'find it'}}]}}
        chunks = target_chunks(record, tokenizer, 5)
        end = 3 + len(chunks) - 1
        schedule = dict(targets=[dict(onset_seconds=3, end_seconds=end)])
        marker_count = 0
        for offset, (ids, kinds) in enumerate(chunks):
            roles, events = supervision_metadata(schedule, 3 + offset, ids, kinds, tokenizer)
            self.assertEqual(roles[0], 3 if offset == 0 else 5)
            marker_count += roles.count(7)
            self.assertTrue(all(e == 0 for e, k in zip(events, kinds) if k))
        self.assertEqual(marker_count, len(tokenizer.encode('<|task_start|>', add_special_tokens=False)))
        for second, role in ((2, 1), (end+1, 2)):
            self.assertEqual(supervision_metadata(schedule, second, [302], [1], tokenizer), ([role], [-1]))
        self.assertEqual(supervision_metadata(schedule, 1, [302], [0], tokenizer), ([0], [-1]))
