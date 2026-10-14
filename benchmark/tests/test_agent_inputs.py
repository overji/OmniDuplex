import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.adapters import joyai, livestar, minicpmo, mmduet2
from benchmark.runtime.runner import SampleContext


class AgentInputTests(unittest.TestCase):
    def test_minicpmo_ignores_backend_results_and_generates_after_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {'mode': 'agent', 'task': 'test', 'video': 'clip.mp4', 'question': 'Q',
                      'question_time': '00:00', 'agent_results': [
                          {'timestamp': '00:01', 'at_seconds': 1, 'text': 'known result',
                           'source_response_index': 1}]}
            context = SampleContext(0, sample, 'Base', root)
            calls = []
            replies = iter([{'text': '', 'is_listen': True, 'end_of_turn': True},
                            {'text': 'reported', 'is_listen': False, 'end_of_turn': True}])

            def prefill(**kwargs):
                calls.append(kwargs)
                return {'success': True}

            class Image:
                def __enter__(self): return self
                def __exit__(self, *unused): pass
                def convert(self, mode): return 'image'

            import sys
            modules = {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext),
                       'numpy': SimpleNamespace(),
                       'transformers': SimpleNamespace(set_seed=lambda seed: None),
                       'PIL': SimpleNamespace(Image=SimpleNamespace(open=lambda path: Image()))}
            metadata = {'frames': ['0.jpg', '1.jpg'], 'frame_times': [0, 1], 'duration': 2}
            args = SimpleNamespace(seed=1, videos_dir=root, cache_dir=root, max_seconds=0,
                                   decode_mode='sampling', max_speak_tokens=20)
            def generate(**kwargs):
                self.assertIn('audio_waveform', calls[-1])
                self.assertIn('frame_list', calls[-1])
                return next(replies)
            model = SimpleNamespace(prepare=lambda **kw: None, streaming_prefill=prefill,
                                    streaming_generate=generate)
            with patch.dict(sys.modules, modules), patch.object(minicpmo, 'media_cache',
                    return_value=(root, metadata, [0.] * 32000)), contextlib.redirect_stdout(io.StringIO()):
                result = minicpmo.run_sample(args, SimpleNamespace(model=model), context)
            self.assertEqual(len(calls), 3)
            self.assertIn('User request', calls[0]['text_list'][0])
            self.assertIn('audio_waveform', calls[1])
            self.assertIn('audio_waveform', calls[2])
            self.assertNotIn('known result', str(calls))
            self.assertEqual(result.events[0]['onset_seconds'], 2)
            self.assertFalse((root / 'sample_000.agent_inputs.json').exists())

    def test_livestar_ignores_backend_results_and_preserves_post_frame_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {'mode': 'agent', 'video': 'clip.mp4', 'question': 'Q',
                      'question_time': '00:00', 'agent_results': [
                          {'timestamp': '00:01', 'at_seconds': 1, 'text': 'known result',
                           'source_response_index': 1}]}
            context = SampleContext(0, sample, 'Base', root)
            questions = []
            def chat(tokenizer, pixels, question, *args, **kwargs):
                questions.append(question)
                return 'reported'
            model = SimpleNamespace(system_message='', chat=chat)

            class Image:
                def __enter__(self): return self
                def __exit__(self, *unused): pass
                def convert(self, mode): return self

            class Session:
                def __init__(self, *args): self.narration = [(1, 'reported')]
                def run(self, length):
                    for frame in [0, 1, 1, 2]:
                        model.chat(None, None, f'Frame-{frame}: <image>')
                    return 'summary'

            native = SimpleNamespace(build_transform=lambda **kw: lambda tile: tile,
                                     dynamic_preprocess=lambda image, **kw: [image],
                                     StreamingSession=Session)
            state = SimpleNamespace(torch=SimpleNamespace(inference_mode=contextlib.nullcontext,
                                                           stack=lambda tiles: tiles),
                                    Image=SimpleNamespace(open=lambda path: Image()), native=native,
                                    model=model, tokenizer=None, gen_cfg={})
            metadata = {'frames': ['0.jpg', '1.jpg', '2.jpg'], 'duration': 3}
            args = SimpleNamespace(videos_dir=root, cache_dir=root)
            with patch.object(livestar, 'media_cache', return_value=(root, metadata, None)):
                result = livestar.run_sample(args, state, context)
            for question in questions:
                self.assertNotIn('known result', question)
            self.assertEqual(result.events[0]['onset_seconds'], 2)
            self.assertIs(model.chat, chat)
            # The post-frame clock applies to every response, not just Agent results.
            context.sample = {**sample, 'mode': 'rtd', 'agent_results': []}
            with patch.object(livestar, 'media_cache', return_value=(root, metadata, None)):
                result = livestar.run_sample(args, state, context)
            self.assertEqual(result.events[0]['onset_seconds'], 2)
            self.assertIs(model.chat, chat)

    def test_joyai_ignores_backend_results_and_preserves_post_frame_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {'mode': 'agent', 'video': 'clip.mp4', 'question': 'Q',
                      'question_time': '00:00', 'agent_results': [
                          {'timestamp': '00:01', 'at_seconds': 1, 'text': 'known result',
                           'source_response_index': 1}]}
            context = SampleContext(0, sample, 'Base', root)
            calls = []

            def post(url, **kwargs):
                if url.endswith('/reset'):
                    content = ''
                else:
                    calls.append(kwargs['json'])
                    content = '</response> reported' if len(calls) == 2 else '</silence>'
                return SimpleNamespace(raise_for_status=lambda: None,
                    json=lambda: {'choices': [{'message': {'content': content}}]})

            args = SimpleNamespace(videos_dir=root, cache_dir=root,
                                   adapter_url='http://localhost/v1/chat/completions', model='JoyAI')
            metadata = {'frames': ['0.jpg', '1.jpg', '2.jpg'],
                        'frame_times': [0, 1, 2], 'duration': 3}
            with patch.object(joyai, 'media_cache', return_value=(root, metadata, None)):
                result = joyai.run_sample(args, SimpleNamespace(requests=SimpleNamespace(post=post)), context)
            content = calls[1]['messages'][1]['content']
            self.assertEqual([c['type'] for c in content], ['image_url'])
            self.assertNotIn('known result', json.dumps(calls))
            self.assertEqual(result.events, [{'onset_seconds': 2, 'text': 'reported'}])
            saved = json.loads((root / 'sample_000.json').read_text())
            self.assertNotIn('agent_inputs', saved)
            self.assertEqual(saved['turns'][1]['frame_time_seconds'], 1)

    def test_mmduet_ignores_backend_results_and_only_generates_after_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = {'mode': 'agent', 'video': 'clip.mp4', 'question': 'Q',
                      'question_time': '00:00', 'agent_results': [
                          {'timestamp': '00:01', 'at_seconds': 1, 'text': 'known result',
                           'source_response_index': 1}]}
            context = SampleContext(0, sample, 'Base', root)
            conversations = []

            class Client:
                def __init__(self, **kwargs): pass
                def set_fps(self, **kwargs): pass
                def input_query_stream(self, conversation): conversations.extend(conversation)
                def inference(self):
                    return {'conversation': [{'role': 'assistant', 'content': 'reported', 'time': 2}]}

            class Image:
                def __enter__(self): return self
                def __exit__(self, *unused): pass
                def convert(self, mode): return 'image'

            metadata = {'frames': ['0.jpg', '1.jpg'], 'frame_times': [0, 1], 'duration': 2}
            import sys
            modules = {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext),
                       'PIL': SimpleNamespace(Image=SimpleNamespace(open=lambda path: Image()))}
            args = SimpleNamespace(videos_dir=root, cache_dir=root)
            state = SimpleNamespace(module=SimpleNamespace(ProactiveInferenceClient=Client),
                                    model=None, processor=None)
            with patch.dict(sys.modules, modules), patch.object(mmduet2, 'media_cache',
                    return_value=(root, metadata, None)):
                result = mmduet2.run_sample(args, state, context)
            self.assertNotIn('known result', str(conversations))
            self.assertEqual(sum(turn['role'] == 'user' and isinstance(turn['content'], str)
                                 for turn in conversations), 1)
            for turn in conversations:
                if turn['role'] == 'user' and isinstance(turn['content'], str):
                    self.assertTrue(turn['skip_inference'])
            self.assertEqual(conversations[-1]['content'][0]['text'], 'Frame at 1.000s')
            self.assertEqual(result.events[0]['onset_seconds'], 2)


if __name__ == '__main__':
    unittest.main()
