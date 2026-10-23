import copy
import json
from pathlib import Path
import tempfile
import unittest

from train.src.data import first_record, target_chunks, prepare


class Tokenizer:
    def encode(self, text, **kw):
        return list(text.encode())
    def convert_tokens_to_ids(self, token):
        return {'<|speak|>': 1000, '<|turn_eos|>': 1001, '<|chunk_eos|>': 1002}[token]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'v.mp4').touch()
        self.config = {'videos_dir': str(self.root), 'max_prefix_seconds': 32, 'seed': 42}
        self.sample = {'mode': 'pr', 'video': 'v.mp4', 'question_time': '00:02',
            'responses': [{'response': 'An event happened.', 'trigger_time': '00:04',
                           'window_start': '00:03', 'window_end': '00:06',
                           'evidence': [{'available_time': '00:05'}]}]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_causal_target_and_unreachable_window(self):
        row, error = first_record(self.sample, 0, self.config)
        self.assertIsNone(error); self.assertEqual(row['onset_seconds'], 5)
        self.sample['responses'][0]['window_end'] = '00:04'
        self.assertEqual(first_record(self.sample, 0, self.config)[1], 'no_causal_integer_onset')

    def test_missing_media_and_no_silent_prefix_truncation(self):
        self.sample['video'] = 'missing.mp4'
        self.assertEqual(first_record(self.sample, 0, self.config)[1], 'missing_video')
        self.sample['video'] = 'v.mp4'; self.config['max_prefix_seconds'] = 3
        self.assertEqual(first_record(self.sample, 0, self.config)[1], 'prefix_too_long')

    def test_reply_spans_chunks_without_losing_text(self):
        record, _ = first_record(self.sample, 0, self.config)
        chunks = target_chunks(record, Tokenizer(), 10)
        self.assertTrue(all(len(ids) <= 10 for ids, _ in chunks))
        content = bytes(t for ids, kinds in chunks for t, k in zip(ids, kinds) if k == 2).decode()
        self.assertEqual(content, record['reference']['response'])
        self.assertEqual(sum(1001 in ids for ids, _ in chunks), 1)
        self.assertTrue(all(kinds[-1] == 0 for _, kinds in chunks))

    def test_agent_serialization_and_tool_result_exclusion(self):
        sample = copy.deepcopy(self.sample)
        sample['mode'] = 'agent'; sample['sample_type'] = 'positive'
        ref = sample['responses'][0]
        ref['role'] = 'assistant'
        ref['tool_calls'] = [{'type': 'function', 'function': {'name': 'agent_call',
            'arguments': '{"instruction":"Look up the object"}'}}]
        sample['responses'].append({'role': 'tool', 'response': 'future secret',
                                    'window_start': '00:07', 'window_end': '00:09'})
        row, _ = first_record(sample, 0, self.config)
        from train.tests.test_agent_token import ByteTokenizer
        chunks = target_chunks(row, ByteTokenizer(), 10)
        content = bytes(t for ids, kinds in chunks for t, k in zip(ids, kinds) if k == 3).decode()
        obj = json.loads(content)
        self.assertEqual(obj, {'name':'task_start','arguments':{'instruction':'Look up the object'}})
        self.assertNotIn('future secret', content)
        self.assertTrue(any(3 in kinds for _, kinds in chunks))

    def test_video_split_and_reproducible_selection(self):
        samples = []
        for i in range(10):
            (self.root / f'{i}.mp4').touch()
            for _ in range(2):
                item = copy.deepcopy(self.sample); item['video'] = f'{i}.mp4'; samples.append(item)
        data = self.root / 'qa.json'; data.write_text(json.dumps(samples))
        prepare(data, self.root / 'a', self.config, 8, .2)
        prepare(data, self.root / 'b', self.config, 8, .2)
        self.assertEqual((self.root / 'a/train.json').read_bytes(), (self.root / 'b/train.json').read_bytes())
        manifest = json.loads((self.root / 'a/manifest.json').read_text())
        self.assertFalse(set(manifest['videos']['train']) & set(manifest['videos']['validation']))

    def test_grpo_excludes_whole_qa_when_a_later_window_exceeds_horizon(self):
        config = dict(self.config, training_method='full_grpo', max_prefix_seconds=32)
        samples = []
        for i in range(10):
            (self.root / f'{i}.mp4').touch()
            item = copy.deepcopy(self.sample); item['video'] = f'{i}.mp4'
            samples.append(item)
            long_item = copy.deepcopy(item)
            long_item['responses'].append(dict(window_start='00:31', window_end='00:33', response='later'))
            samples.append(long_item)
        data = self.root / 'grpo.json'; data.write_text(json.dumps(samples))
        prepare(data, self.root / 'grpo', config, 8, .2)
        manifest = json.loads((self.root / 'grpo/manifest.json').read_text())
        self.assertEqual(manifest['skipped']['window_exceeds_rollout_horizon'], 10)
        for split in ('train', 'validation'):
            self.assertTrue(all(len(r['sample']['responses']) == 1 for r in json.loads((self.root / f'grpo/{split}.json').read_text())))

if __name__ == '__main__':
    unittest.main()
