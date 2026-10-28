"""Remote exports must map by QA identity, never by split-local indices."""
import copy
import unittest

from benchmark.scripts.merge_remote_probe_results import normalize, read_records
from benchmark.probe.data import SCHEMA, digest


class RemoteProbeMergeTests(unittest.TestCase):
    def setUp(self):
        self.canonical = [{'question': 'first'}, {'question': 'second'}]
        self.samples = list(reversed(self.canonical))
        self.row = {'sample_index': 1, 'response_index': 0, 'side': 'post',
                    'window_start_seconds': 10, 'window_end_seconds': 12,
                    'pair_id': 'canonical-pair', 'probe_id': 'canonical-pair:post'}
        pair = digest([SCHEMA, 'split-hash', 0, 0, 9, 12])
        self.prediction = {'sample_index': 0, 'response_index': 0, 'side': 'post',
                           'pair_id': pair, 'probe_id': pair + ':post', 'status': 'complete',
                           'output_protocol': 'probe_reply_task_text_v1',
                           'raw_response': 'YES\nHello<|task_start|>Check the weather'}

    def normalize(self, records):
        return normalize(records, self.samples, 'split-hash', self.canonical,
                         [self.row], 'remaining1607', 'archive!predictions.jsonl')

    def test_reordered_split_preserves_payload_and_original_identity(self):
        before = copy.deepcopy(self.prediction)
        records, mapping = self.normalize([self.prediction])
        self.assertEqual(mapping, {0: 1, 1: 0})
        self.assertEqual(records[0]['sample_index'], 1)
        self.assertEqual(records[0]['probe_id'], 'canonical-pair:post')
        self.assertEqual(records[0]['raw_response'], before['raw_response'])
        self.assertEqual(records[0]['_merge_source']['probe_id'], before['probe_id'])
        self.assertEqual(self.prediction, before)

    def test_rejects_duplicates_and_changed_window_identity(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate remote'):
            self.normalize([self.prediction, self.prediction])
        changed = {**self.prediction, 'pair_id': 'wrong-window'}
        with self.assertRaisesRegex(ValueError, 'prefix-window identity'):
            self.normalize([changed])

    def test_rejects_unknown_qa_and_incomplete_output(self):
        with self.assertRaisesRegex(ValueError, 'incomplete remote'):
            self.normalize([{**self.prediction, 'status': 'error'}])
        with self.assertRaisesRegex(ValueError, 'no canonical'):
            normalize([], [{'question': 'unknown'}], 'split-hash', self.canonical,
                      [self.row], 'remaining1607', 'archive')

    def test_rejects_interrupted_jsonl(self):
        with self.assertRaisesRegex(ValueError, 'interrupted final line'):
            read_records(b'{"status":"complete"}')


if __name__ == '__main__':
    unittest.main()
