"""Gemini 2.5 Flash Live using native audio output and its output transcript."""
import time
from types import SimpleNamespace

from benchmark.adapters import cloud_realtime as cloud


STRICT_CONFIG = True
SELECT_BEFORE_RESUME = True


def run_config(args, base):
    return cloud.run_config(args, base, 'gemini_live')


def load(args):
    from google import genai
    from google.genai import types
    return SimpleNamespace(genai=genai, types=types, api_key=cloud.api_key(args))


class GeminiSession:
    def __init__(self, args, state, log):
        self.args, self.state, self.log = args, state, log
        self.client = self.manager = self.session = None
        self.activity_open = False

    async def open(self, system):
        self.client = self.state.genai.Client(api_key=self.state.api_key, vertexai=False,
            http_options={'api_version': self.args.api_version,
                          'base_url': cloud.endpoint(self.args, 'gemini_live'),
                          'timeout': round(self.args.api_timeout * 1000)})
        config = {'response_modalities': ['AUDIO'], 'output_audio_transcription': {},
                  'system_instruction': system,
                  'realtime_input_config': {'automatic_activity_detection': {'disabled': True}}}
        if self.args.max_output_tokens is not None:
            config['max_output_tokens'] = self.args.max_output_tokens
        self.manager = self.client.aio.live.connect(model=self.args.model, config=config)
        self.session = await self.manager.__aenter__()
        self.log('server', {'type': 'setup_complete'})

    async def step(self, step, question, respond):
        types = self.state.types
        if not self.activity_open:
            await self.session.send_realtime_input(activity_start=types.ActivityStart())
            self.activity_open = True
        if question is not None:
            # Keep text on the same realtime channel as media, inside the activity.
            await self.session.send_realtime_input(text=question)
        if step['pcm'] is not None:
            await self.session.send_realtime_input(
                audio=types.Blob(data=step['pcm'], mime_type='audio/pcm;rate=16000'))
        for _, jpeg in step['frames']:
            await self.session.send_realtime_input(
                video=types.Blob(data=jpeg, mime_type='image/jpeg'))
        if not respond:
            # Prefix evidence remains in one open activity until the request arrives.
            return None
        await self.session.send_realtime_input(activity_end=types.ActivityEnd())
        self.activity_open = False
        transcript, text_parts = [], []
        audio_bytes = 0
        first = None
        usage = None
        completed = False
        async for message in self.session.receive():
            if message.go_away is not None:
                raise RuntimeError('Gemini requested session shutdown; this sample was not completed')
            if message.tool_call is not None:
                raise RuntimeError('Unexpected native tool call; benchmark does not execute tools')
            if message.usage_metadata is not None:
                usage = message.usage_metadata.model_dump(mode='json', exclude_none=True)
            content = message.server_content
            if content is None:
                continue
            if content.interrupted:
                raise RuntimeError('Gemini interrupted a media step; this sample was not completed')
            transcription = content.output_transcription
            delta = transcription.text if transcription is not None else None
            if delta:
                transcript.append(delta)
            parts = content.model_turn.parts if content.model_turn is not None else []
            for part in parts or []:
                if part.inline_data is not None:
                    audio_bytes += len(part.inline_data.data or b'')
                if part.text and not part.thought:
                    text_parts.append(part.text)
            if first is None and (delta or audio_bytes or text_parts):
                first = time.perf_counter()
            self.log('server', {'type': 'server_content', 'output_transcript_delta': delta,
                'text_parts': [p.text for p in parts or [] if p.text and not p.thought],
                'audio_bytes_so_far': audio_bytes, 'turn_complete': bool(content.turn_complete)})
            if content.turn_complete:
                completed = True
                break
        if not completed:
            raise RuntimeError('Gemini stream ended before turn_complete')
        text = ''.join(transcript) or ''.join(text_parts)
        if audio_bytes and not text.strip():
            raise RuntimeError('Gemini returned audio without output transcription; cannot score its content')
        return {'text': text, 'usage': usage, 'generated_audio_bytes': audio_bytes,
                'first_output_monotonic': first}

    async def close(self):
        try:
            if self.session is not None:
                await self.manager.__aexit__(None, None, None)
        finally:
            if self.client is not None:
                await self.client.aio.aclose()


def run_sample(args, state, context):
    return cloud.run_sample(args, state, context, GeminiSession)
