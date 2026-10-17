"""Offline protocol checks; no credentials or external API calls are required."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from benchmark.adapters import cloud_realtime as cloud, qwen_realtime, gemini_live
from benchmark.cli.main import build_parser
from benchmark.runtime.runner import SampleContext


class CloudConfigTests(unittest.TestCase):
    def test_cloud_models_need_no_weights_or_gpu_arguments(self):
        parser = build_parser()
        qwen = parser.parse_args(['run', 'qwen_realtime'])
        self.assertEqual(qwen.model, 'qwen3.5-omni-plus-realtime')
        flash = parser.parse_args(['run', 'qwen_realtime', '--model', 'qwen3.5-omni-flash-realtime'])
        self.assertEqual(flash.model, 'qwen3.5-omni-flash-realtime')
        gemini = parser.parse_args(['run', 'gemini_live'])
        self.assertIn('gemini-2.5-flash', gemini.model)
        self.assertFalse(hasattr(gemini, 'model_path'))
        self.assertFalse(hasattr(gemini, 'gpus'))

    def test_config_fingerprints_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(['run', 'qwen_realtime', '--base-url', 'wss://example.test/realtime'])
            args.data = Path(tmp) / 'data.json'
            args.data.write_text('[]')
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'secret-key'}):
                config = qwen_realtime.run_config(args, 'system')
            self.assertNotIn('secret-key', json.dumps(config))
            self.assertEqual(config['turn_control'], 'manual_commit_then_response_create_per_media_second')
            self.assertIn('protocol_sha256', config)
            args.base_url = 'wss://example.test/realtime?api_key=secret'
            with self.assertRaises(ValueError):
                qwen_realtime.run_config(args, 'system')

    def test_frame_at_end_is_not_visible_until_next_step(self):
        args = NS(max_seconds=2)
        meta = {'duration': 3., 'has_audio': False,
                'frame_times': [0., 1., 2.], 'frames': ['0.jpg', '1.jpg', '2.jpg']}
        with patch.object(cloud, 'jpeg_frame', side_effect=lambda path, _: path.name.encode()):
            steps = list(cloud.media_steps(Path('/cache'), meta, None, args))
        self.assertEqual([[at for at, _ in step['frames']] for step in steps], [[0.], [1.]])
        self.assertEqual([step['end'] for step in steps], [1., 2.])
        self.assertTrue(all(step['pcm'] is None for step in steps))

    def test_listen_marker_does_not_remove_real_response_or_structured_json(self):
        for text in ('SILENCE', '<silence>', 'Silence.'):
            self.assertEqual(cloud.spoken_text(text), '')
        for text in ('A silence follows the music.', '{"response":"SILENCE"}', '保持 SILENCE'):
            self.assertEqual(cloud.spoken_text(text), text)


class CloudProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_loop_inserts_question_once_and_scores_after_media(self):
        calls = []
        class Session:
            def __init__(self, *unused):
                pass

            async def open(self, system):
                calls.append(('open', system))

            async def step(self, step, question, respond):
                calls.append(('step', step['start'], step['end'], question, respond))
                if not respond:
                    return None
                return {'text': 'SILENCE' if step['start'] == 1 else 'The cup moved.',
                        'first_output_monotonic': time.perf_counter()}

            async def close(self):
                calls.append(('close',))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = NS(videos_dir=root, cache_dir=root, max_seconds=0, api_timeout=5)
            sample = {'video': 'test.mp4', 'question': 'Tell me when the cup moves.',
                      'question_time': '00:01', 'available_tools': [], 'mode': 'pr'}
            context = SampleContext(0, sample, 'Base', root, started=time.perf_counter())
            steps = [{'start': float(i), 'end': float(i + 1), 'frames': [], 'pcm': None}
                     for i in range(3)]
            with patch.object(cloud, 'media_cache', return_value=(root, {'duration': 3, 'has_audio': False}, None)), \
                    patch.object(cloud, 'media_steps', return_value=steps):
                result = await cloud.stream_sample(args, None, context, Session)
            media_calls = [call for call in calls if call[0] == 'step']
            self.assertIsNone(media_calls[0][3])
            self.assertFalse(media_calls[0][4])
            self.assertIn(sample['question'], media_calls[1][3])
            self.assertIsNone(media_calls[2][3])
            self.assertEqual(result.events[0]['onset_seconds'], 3.)
            self.assertEqual(len(result.events), 1)
            self.assertEqual(calls[-1], ('close',))

    async def test_qwen_commits_media_before_response_and_does_not_duplicate_final_text(self):
        replies = [
            {'type': 'session.created'}, {'type': 'session.updated'},
            {'type': 'input_audio_buffer.committed'},
            {'type': 'response.text.delta', 'delta': 'Hello'},
            {'type': 'response.text.done', 'text': 'Hello'},
            {'type': 'response.done', 'response': {'status': 'completed', 'usage': {'output_tokens': 1}}},
        ]
        ws = NS(send=AsyncMock(), recv=AsyncMock(side_effect=[json.dumps(r) for r in replies]), close=AsyncMock())
        args = NS(base_url='wss://example.test/realtime', model='qwen3.5-omni-flash-realtime',
                  api_timeout=5, max_output_tokens=None)
        state = NS(connect=AsyncMock(return_value=ws), api_key='secret')
        session = qwen_realtime.QwenSession(args, state, lambda *unused: None)
        await session.open('system')
        result = await session.step({'start': 0., 'end': 1., 'pcm': b'\0' * 32000,
                                    'frames': [(0., b'jpeg')]}, 'Question', True)
        sent = [json.loads(call.args[0]) for call in ws.send.await_args_list]
        kinds = [event['type'] for event in sent]
        self.assertLess(kinds.index('input_image_buffer.append'), kinds.index('input_audio_buffer.commit'))
        self.assertLess(kinds.index('input_audio_buffer.commit'), kinds.index('response.create'))
        self.assertEqual(sent[0]['session']['turn_detection'], None)
        self.assertEqual(result['text'], 'Hello')
        await session.close()

    async def test_gemini_activity_boundary_and_output_transcript(self):
        async def receive():
            yield NS(go_away=None, tool_call=None, usage_metadata=None, server_content=NS(
                interrupted=False, output_transcription=NS(text='The cup moved.'),
                model_turn=None, turn_complete=True))
        native = NS(send_realtime_input=AsyncMock(), receive=receive)
        state = NS(types=NS(ActivityStart=lambda: 'start', ActivityEnd=lambda: 'end',
                            Blob=lambda **kwargs: kwargs))
        session = gemini_live.GeminiSession(NS(), state, lambda *unused: None)
        session.session = native
        step = {'pcm': None, 'frames': [(0., b'jpeg')]}
        self.assertIsNone(await session.step(step, None, False))
        result = await session.step(step, 'Question', True)
        sent = [call.kwargs for call in native.send_realtime_input.await_args_list]
        self.assertEqual(sent[0], {'activity_start': 'start'})
        self.assertEqual(sent[-1], {'activity_end': 'end'})
        self.assertTrue(all('audio' not in packet for packet in sent))
        self.assertEqual(result['text'], 'The cup moved.')
