"""Qwen3.5-Omni Plus/Flash Realtime via the native DashScope WebSocket API."""
import base64
import json
import time
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from benchmark.adapters import cloud_realtime as cloud


STRICT_CONFIG = True
SELECT_BEFORE_RESUME = True


def run_config(args, base):
    return cloud.run_config(args, base, 'qwen_realtime')


def load(args):
    from websockets.asyncio.client import connect
    return SimpleNamespace(connect=connect, api_key=cloud.api_key(args))


def model_url(endpoint, model):
    parsed = urlsplit(endpoint)
    query = dict(parse_qsl(parsed.query))
    query['model'] = model
    return urlunsplit(parsed._replace(query=urlencode(query)))


class QwenSession:
    def __init__(self, args, state, log):
        self.args, self.state, self.log = args, state, log
        self.ws = None

    async def send(self, event):
        # Never log Authorization headers or Base64 media.
        self.log('client', {'type': event['type']})
        await self.ws.send(json.dumps(event))

    async def receive(self):
        event = json.loads(await self.ws.recv())
        self.log('server', {key: event[key] for key in
            ('type', 'event_id', 'response_id', 'text', 'delta', 'error') if key in event
            and not (key == 'delta' and event['type'] == 'response.audio.delta')})
        if event.get('type') == 'error':
            raise RuntimeError('DashScope Realtime error: ' + json.dumps(event.get('error', {})))
        return event

    async def wait_for(self, kind):
        while True:
            event = await self.receive()
            if event['type'] == kind:
                return event

    async def open(self, system):
        url = model_url(cloud.endpoint(self.args, 'qwen_realtime'), self.args.model)
        self.ws = await self.state.connect(url,
            additional_headers={'Authorization': 'Bearer ' + self.state.api_key},
            open_timeout=self.args.api_timeout, close_timeout=10, max_size=8 * 1024 * 1024)
        await self.wait_for('session.created')
        config = {'modalities': ['text'], 'instructions': system,
                  'input_audio_format': 'pcm', 'input_audio_transcription': None,
                  'turn_detection': None, 'enable_search': False}
        if self.args.max_output_tokens is not None:
            config['max_tokens'] = self.args.max_output_tokens
        await self.send({'type': 'session.update', 'session': config})
        await self.wait_for('session.updated')

    async def step(self, step, question, respond):
        if question is not None:
            await self.send({'type': 'conversation.item.create', 'item': {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': question}]}})
        pcm = step['pcm']
        if pcm is None:
            # Image buffers require an audio append before each image/commit.
            pcm = bytes(round((step['end'] - step['start']) * 16000) * 2)
            self.log('input', {'type': 'synthetic_silent_audio', 'bytes': len(pcm)})
        if len(pcm) < 3200:
            # A trailing fractional second can be shorter than the commit minimum.
            self.log('input', {'type': 'audio_commit_padding', 'bytes': 3200 - len(pcm)})
            pcm += bytes(3200 - len(pcm))
        await self.send({'type': 'input_audio_buffer.append',
                         'audio': base64.b64encode(pcm).decode('ascii')})
        for _, jpeg in step['frames']:
            await self.send({'type': 'input_image_buffer.append',
                             'image': base64.b64encode(jpeg).decode('ascii')})
        await self.send({'type': 'input_audio_buffer.commit'})
        await self.wait_for('input_audio_buffer.committed')
        if not respond:
            return None
        await self.send({'type': 'response.create'})
        text = ''
        first = None
        while True:
            event = await self.receive()
            kind = event['type']
            if kind == 'response.text.delta':
                delta = event.get('delta', '')
                if delta and first is None:
                    first = time.perf_counter()
                text += delta
            elif kind == 'response.text.done':
                text = event.get('text', text)
                if text and first is None:
                    first = time.perf_counter()
            elif kind == 'response.done':
                response = event.get('response', {})
                self.log('server', {'type': 'response_summary',
                    'status': response.get('status'), 'usage': response.get('usage')})
                if response.get('status') not in (None, 'completed'):
                    raise RuntimeError('DashScope response did not complete: ' +
                                       json.dumps(response.get('status_details', response.get('status'))))
                if not text:
                    text = ''.join(part.get('text', '') for item in response.get('output', [])
                                   for part in item.get('content', []))
                    if text and first is None:
                        first = time.perf_counter()
                calls = [{'type': 'function', 'function': {'name': item.get('name'),
                          'arguments': item.get('arguments')}} for item in response.get('output', [])
                         if item.get('type') == 'function_call']
                result = {'text': text, 'usage': response.get('usage'),
                          'first_output_monotonic': first}
                if calls:
                    result['tool_calls'] = calls
                return result

    async def close(self):
        if self.ws is not None:
            await self.ws.close()


def run_sample(args, state, context):
    return cloud.run_sample(args, state, context, QwenSession)
