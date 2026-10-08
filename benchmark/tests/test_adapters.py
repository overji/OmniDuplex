import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark.adapters import joyai, minicpmo
from benchmark.runtime.runner import SampleContext


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample = dict(task='test', video='test.mp4', question='Notify me',
                           question_time='00:01', available_tools=[])
        self.context = SampleContext(0, self.sample, 'Base prompt', self.root)
        self.metadata = dict(duration=2.5, frame_times=[0.0, 1.0, 2.0],
                             frames=['0.jpg', '1.jpg', '2.jpg'])

    def test_minicpmo_causal_frames_question_clock_and_chunk_merging(self):
        calls = []
        frames = []

        class Image:
            def __init__(self, path):
                frames.append(Path(path).name)

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pass

            def convert(self, mode):
                return self

        class Audio:
            def __getitem__(self, region):
                return [0.0] * (region.stop - region.start)

        replies = iter([
            dict(text='', is_listen=True, end_of_turn=True),
            dict(text='A', is_listen=False, end_of_turn=False),
            dict(text='B', is_listen=False, end_of_turn=True),
            dict(text='', is_listen=True, end_of_turn=True),
        ])
        def prefill(**kwargs):
            calls.append(kwargs)
            return {'success': True}

        model = SimpleNamespace(prepare=Mock(), streaming_prefill=prefill,
                                streaming_generate=lambda **kw: next(replies))
        modules = {
            'torch': SimpleNamespace(inference_mode=contextlib.nullcontext),
            'numpy': SimpleNamespace(pad=lambda values, widths: values + [0.0] * widths[1]),
            'PIL': SimpleNamespace(Image=SimpleNamespace(open=Image)),
            'transformers': SimpleNamespace(set_seed=Mock()),
        }
        args = SimpleNamespace(seed=42, videos_dir=self.root, cache_dir=self.root,
                               max_seconds=0, decode_mode='sampling', max_speak_tokens=20)
        with patch.dict(sys.modules, modules), patch.object(minicpmo, 'media_cache',
                return_value=(self.root, self.metadata, Audio())), \
                contextlib.redirect_stdout(io.StringIO()):
            result = minicpmo.run_sample(args, SimpleNamespace(model=model), self.context)
        self.assertEqual(frames, ['0.jpg', '1.jpg', '2.jpg'])
        self.assertIn('audio_waveform', calls[0])
        self.assertIn('text_list', calls[1])
        self.assertEqual(len(calls[-1]['audio_waveform']), 16000)
        self.assertEqual(result.events, [dict(onset_seconds=1.0, text='AB', last_text_seconds=2.0)])
        self.assertEqual(result.observed_seconds, 2.5)
        self.assertFalse(result.include_sample_type)
        raw = [json.loads(line) for line in
               (self.root / 'sample_000.chunks.jsonl').read_text().splitlines()]
        self.assertEqual([entry['video_time_seconds'] for entry in raw], [1.0, 1.0, 2.0, 2.5])

    def test_joyai_native_silence_reset_and_question_injection(self):
        calls = []
        outputs = iter(['</response> Before question', '</silence>', '</response> Reply'])

        def post(url, **kwargs):
            calls.append((url, kwargs))
            content = '' if url.endswith('/reset') else next(outputs)
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'choices': [{'message': {'content': content}}]})

        args = SimpleNamespace(videos_dir=self.root, cache_dir=self.root,
                               adapter_url='http://localhost/v1/chat/completions', model='JoyAI')
        with patch.object(joyai, 'media_cache', return_value=(self.root, self.metadata, None)):
            result = joyai.run_sample(args, SimpleNamespace(requests=SimpleNamespace(post=post)),
                                      self.context)
        self.assertEqual(calls[0][1]['json'], {})
        contents = [call[1]['json']['messages'][1]['content'] for call in calls[1:]]
        self.assertEqual([sum(item['type'] == 'text' for item in content)
                          for content in contents], [0, 1, 0])
        self.assertEqual(result.events, [dict(onset_seconds=2.0, text='Reply')])
        self.assertEqual(len(json.loads((self.root / 'sample_000.json').read_text())['turns']), 3)


if __name__ == '__main__':
    unittest.main()
