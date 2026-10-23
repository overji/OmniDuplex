"""Lossless Agent protocol parsing and post-media event collection."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from benchmark.adapters import minicpmo, gander
from benchmark.evaluation.agent_calls import parse_event, native_event
from benchmark.runtime.runner import SampleContext


CALL = {'name': 'agent_call', 'arguments': {'instruction': 'Save the observed item'}}


class AgentProtocolTests(unittest.TestCase):
    def test_native_and_openai_shapes_have_same_normalized_call(self):
        expected = parse_event({'text': json.dumps({'tool_calls': [
            {'type': 'function', 'function': CALL}]})})['calls']
        for event in ({'text': json.dumps(CALL)}, {'text': '', 'tool_calls': [CALL]},
                      {'text': '<tool_call>\n' + json.dumps(CALL) + '\n</tool_call>'}):
            parsed = parse_event(event)
            self.assertEqual(parsed['calls'], expected)
            self.assertEqual(parsed['status'], 'structured')

    def test_plain_promises_and_truncated_calls_are_not_recovered(self):
        self.assertEqual(parse_event({'text': 'I will call agent_call now'})['calls'], [])
        parsed = parse_event({'text': '<tool_call>{"name":"agent_call",'})
        self.assertEqual(parsed['status'], 'malformed')
        self.assertEqual(parsed['calls'], [])
        malformed = native_event({'tool_calls': [], 'tool_error': 'invalid schema',
                                  'raw_tool_text': '<tool_call>broken'}, 4.)
        self.assertTrue(parse_event(malformed)['attempted'])
        self.assertEqual(parse_event(malformed)['status'], 'malformed')

    def test_native_runner_metadata_is_not_concatenated_into_speech(self):
        result = {'text': '', 'is_tool_call': True, 'tool_calls': [CALL],
                  'tool_generation_complete': True, 'tool_parse_valid': True,
                  'tool_schema_valid': True, 'end_of_turn': False, 'is_listen': False}
        wrapper = gander.StreamingThinker(None, None, None)
        wrapper.runner = NS(streaming_generate=Mock(return_value=result))
        self.assertIs(wrapper.streaming_generate(), result)
        event = native_event(result, 5., {'onset_seconds': 4., 'text': 'I will save it.'})
        self.assertEqual(event['response_text'], 'I will save it.')
        self.assertEqual(event['onset_seconds'], 4.)
        self.assertEqual(event['agent_call_onset_seconds'], 5.)
        self.assertEqual(parse_event(event)['calls'][0]['function'], CALL)

    def run_stream(self, replies):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = {'mode': 'agent', 'task': 'test_agent', 'video': 'v.mp4', 'question': 'Save it when it appears',
                      'question_time': '00:00', 'frontend_context': {}, 'available_tools': []}
            context = SampleContext(0, sample, 'Base', root, agent_output_protocol='structured')
            model = NS(prepare=Mock(), streaming_prefill=Mock(return_value={'success': True}),
                       streaming_generate=Mock(side_effect=replies))
            args = NS(seed=1, videos_dir=root, cache_dir=root, max_seconds=0,
                      decode_mode='greedy', max_speak_tokens=20)
            meta = {'frames': [], 'frame_times': [], 'duration': len(replies)}
            with patch.object(minicpmo, 'media_cache', return_value=(root, meta, np.zeros(len(replies)*16000))), contextlib.redirect_stdout(io.StringIO()):
                result = minicpmo.run_sample(args, NS(model=model), context)
            self.assertEqual(model.streaming_prefill.call_count, len(replies)+1)
            return result.events

    def test_json_split_across_media_chunks_keeps_first_onset_and_full_call(self):
        text = json.dumps({'response': 'Saving it.', 'tool_calls': [{'function': CALL}]})
        split = len(text)//2
        events = self.run_stream([
            {'text': text[:split], 'is_listen': False, 'end_of_turn': False},
            {'text': text[split:], 'is_listen': False, 'end_of_turn': True}])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['text'], text)
        self.assertEqual(events[0]['onset_seconds'], 1.)
        self.assertEqual(parse_event(events[0])['calls'][0]['function'], CALL)

    def test_tool_only_unit_is_saved_even_without_text_or_turn_complete(self):
        events = self.run_stream([{'text': '', 'is_listen': False, 'end_of_turn': False,
                                  'is_tool_call': True, 'tool_calls': [CALL]}])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['onset_seconds'], 1.)
        self.assertEqual(parse_event(events[0])['calls'][0]['function'], CALL)


if __name__ == '__main__':
    unittest.main()

class TransportToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_final_native_function_call_is_retained(self):
        from unittest.mock import AsyncMock
        from benchmark.adapters.qwen_realtime import QwenSession
        replies = [{'type': 'input_audio_buffer.committed'}, {'type': 'response.done',
                    'response': {'status': 'completed', 'output': [
                        {'type': 'function_call', **CALL}]}}]
        ws = NS(send=AsyncMock(), recv=AsyncMock(side_effect=[json.dumps(r) for r in replies]))
        session = QwenSession(NS(), None, lambda *args: None)
        session.ws = ws
        result = await session.step({'start': 0., 'end': 1., 'pcm': bytes(32000), 'frames': []}, 'Request', True)
        self.assertEqual(result['text'], '')
        self.assertEqual(parse_event(result)['calls'][0]['function'], CALL)

    async def test_cloud_collector_saves_tool_only_output(self):
        from benchmark.adapters import cloud_realtime as cloud
        class Session:
            def __init__(self, *args): pass
            async def open(self, system): pass
            async def close(self): pass
            async def step(self, step, question, respond):
                return {'text': '', 'tool_calls': [CALL]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = {'mode': 'agent', 'question': 'Save it', 'question_time': '00:00', 'video': 'v.mp4'}
            context = SampleContext(0, sample, 'Base', root)
            args = NS(videos_dir=root, cache_dir=root, max_seconds=0, api_timeout=5)
            step = {'start': 0., 'end': 1., 'frames': [], 'pcm': None}
            with patch.object(cloud, 'media_cache', return_value=(root, {'duration': 1., 'has_audio': False}, None)), patch.object(cloud, 'media_steps', return_value=[step]):
                result = await cloud.stream_sample(args, None, context, Session)
            self.assertEqual(len(result.events), 1)
            self.assertEqual(result.events[0]['onset_seconds'], 1.)
            self.assertEqual(parse_event(result.events[0])['calls'][0]['function'], CALL)


class JoyAIToolTests(unittest.TestCase):
    def test_native_tool_message_with_null_content_is_retained(self):
        from benchmark.adapters import joyai
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = SampleContext(0, {'mode': 'agent', 'video': 'v.mp4', 'question': 'Save it',
                                    'question_time': '00:00'}, 'Base', root)
            def post(url, **kwargs):
                body = {} if url.endswith('/reset') else {'choices': [{'message': {'content': None, 'tool_calls': [CALL]}}]}
                return NS(raise_for_status=lambda: None, json=lambda: body)
            args = NS(videos_dir=root, cache_dir=root, adapter_url='http://localhost/v1/chat/completions', model='test')
            meta = {'duration': 1., 'frames': ['0.jpg'], 'frame_times': [0.]}
            with patch.object(joyai, 'media_cache', return_value=(root, meta, None)):
                result = joyai.run_sample(args, NS(requests=NS(post=post)), context)
            self.assertEqual(parse_event(result.events[0])['calls'][0]['function'], CALL)
