import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.probe.data import reference_history, file_hash
from benchmark.probe.prompts import model_input
from benchmark.probe.metrics import summarize
from benchmark.probe.runner import score
from benchmark.tests.test_probe import sample, probes


class ReferenceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.sample = sample()
        self.sample['responses'][0]['response'] = 'PAST_REPLY: the cup moved to the table.'
        self.sample['responses'][0]['evidence'] = [{'available_time': '00:11', 'fact': 'PAST_EVIDENCE'}]
        self.sample['responses'][1]['response'] = 'CURRENT_SECRET'
        self.rows = probes(self.sample)[0]

    def test_only_earlier_reply_texts_are_exposed_and_both_sides_match(self):
        before = copy.deepcopy(self.sample)
        rows, excluded = reference_history(self.rows, [self.sample], 'source-hash')
        self.assertEqual(excluded, [])
        self.assertEqual(rows[0]['handled_responses'], [])
        self.assertEqual(rows[2]['handled_responses'], rows[3]['handled_responses'])
        for row in rows[2:]:
            prompt = model_input(row, Path('prefix.mp4'), 'Rules', 'reply-task-text',
                                 'task-specific', 'reference')
            self.assertIn('PAST_REPLY', prompt.user_prompt)
            self.assertNotIn('CURRENT_SECRET', prompt.user_prompt)
            self.assertNotIn('PAST_EVIDENCE', prompt.user_prompt)
            self.assertIn('ground-truth reference history', prompt.user_prompt)
        self.assertEqual(self.sample, before)

    def test_unavailable_prior_facts_exclude_both_sides(self):
        self.sample['responses'][0]['evidence'][0]['available_time'] = '00:20'
        rows, excluded = reference_history(self.rows, [self.sample], 'source-hash')
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(x['reason']=='previous_reference_not_available_before_negative_cutoff'
                            for x in excluded))

    def test_reference_dataset_identity_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            reference_history(self.rows, [self.sample], 'wrong-data')

    def test_reference_mode_requires_complete_history(self):
        with self.assertRaisesRegex(ValueError, 'handled reference'):
            model_input(self.rows[2], Path('prefix.mp4'), 'Rules', history_mode='reference')

    def test_count_mode_ignores_attached_gold_history(self):
        row = copy.deepcopy(self.rows[2])
        original = model_input(row, Path('prefix.mp4'), 'Rules')
        row['handled_responses'] = [{'response': 'PAST_REPLY'}]
        self.assertEqual(original, model_input(row, Path('prefix.mp4'), 'Rules'))

    def test_negative_only_scores_have_no_paired_or_positive_accuracy(self):
        rows, _ = reference_history(self.rows, [self.sample], 'source-hash')
        rows = [r for r in rows if r['side']=='pre']
        predictions = [dict(probe_id=r['probe_id'],status='complete',raw_response='NO') for r in rows]
        score = summarize(rows, predictions)['groups']['overall']
        self.assertEqual(score['negative_accuracy'], 1)
        self.assertEqual(score['probe_coverage'], 1)
        self.assertIsNone(score['positive_accuracy'])
        self.assertIsNone(score['paired_accuracy'])

    def test_rescoring_preserves_single_side_and_saved_history_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            (root/'history_exclusions.json').write_text(json.dumps([
                {'probe_id':r['probe_id']} for r in self.rows[2:]]))
            (root/'probe_config.json').write_text(json.dumps(dict(
                manifest_sha256=file_hash(manifest),history_mode='reference',probe_side='pre')))
            predictions=root/'probe_predictions.jsonl'
            predictions.write_text(json.dumps(dict(probe_id=self.rows[0]['probe_id'],
                status='complete',raw_response='NO'))+'\n')
            score(SimpleNamespace(manifest=manifest,predictions=predictions,output_dir=root/'score'))
            metrics=json.loads((root/'score/probe_metrics.json').read_text())['groups']['overall']
            self.assertEqual(metrics['eligible_probes'],1)
            self.assertEqual(metrics['probe_coverage'],1)
