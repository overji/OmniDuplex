"""Causal one-second media steps shared by cloud Realtime transports."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import re
import time
from urllib.parse import parse_qsl, urlsplit

from benchmark.config.defaults import DEFAULTS
from benchmark.config.models import MODEL_SETTINGS
from benchmark.config.paths import CONFIGS
from benchmark.data.media import media_cache
from benchmark.data.prompts import user_request
from benchmark.data.timestamps import seconds
from benchmark.data.types import SampleResult


PROTOCOL_PROMPT = CONFIGS / 'prompts/cloud_realtime.txt'


def endpoint(args, provider):
    value = args.base_url or os.environ.get(DEFAULTS[provider]['base_url_env'])
    if not value and provider == 'gemini_live':
        value = 'https://generativelanguage.googleapis.com'
    if not value:
        raise ValueError('Set --base-url or ' + DEFAULTS[provider]['base_url_env'])
    parsed = urlsplit(value)
    scheme = 'wss' if provider == 'qwen_realtime' else 'https'
    allowed_query = {'model'} if provider == 'qwen_realtime' else set()
    if (parsed.scheme != scheme or not parsed.hostname or parsed.username or parsed.password
            or parsed.fragment or any(key not in allowed_query for key, _ in parse_qsl(parsed.query))):
        raise ValueError(f'Expected a {scheme} endpoint without credentials or API-key query parameters')
    return value


def run_config(args, base_prompt, provider):
    if (not math.isfinite(args.api_timeout) or args.api_timeout <= 0
            or not math.isfinite(args.max_seconds) or args.max_seconds < 0
            or args.max_samples < 0 or args.sample_start < 0 or args.sample_step < 1
            or not 1 <= args.jpeg_quality <= 95 or not 1 <= args.frame_max_edge <= 1080
            or (args.max_output_tokens is not None and args.max_output_tokens < 1)):
        raise ValueError('Invalid timeout, sample range, media limit or output-token limit')
    return {
        **MODEL_SETTINGS[provider], 'adapter': provider, 'model': args.model,
        'base_url': endpoint(args, provider), 'api_version': getattr(args, 'api_version', None),
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'prompt_sha256': hashlib.sha256(base_prompt.encode()).hexdigest(),
        'protocol_sha256': hashlib.sha256(PROTOCOL_PROMPT.read_bytes()).hexdigest(),
        'videos_dir': str(args.videos_dir.resolve()),
        'api_key_env': args.api_key_env, 'api_timeout': args.api_timeout,
        'max_seconds': args.max_seconds, 'max_samples': args.max_samples,
        'sample_start': args.sample_start, 'sample_step': args.sample_step,
        'frame_max_edge': args.frame_max_edge, 'jpeg_quality': args.jpeg_quality,
        'max_output_tokens': args.max_output_tokens,
        'input_pacing': 'at_most_one_media_step_per_wall_second_then_wait_for_response',
        'agent_output_protocol': getattr(args, 'agent_output_protocol', 'text'),
    }


def api_key(args):
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError(f'Set {args.api_key_env} before running this cloud adapter')
    return key


def jpeg_frame(path, args):
    from PIL import Image
    with Image.open(path) as original:
        image = original.convert('RGB')
        image.thumbnail((args.frame_max_edge, args.frame_max_edge))
        while True:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=args.jpeg_quality)
            value = buffer.getvalue()
            # DashScope requires Base64 <=256 KB; use a conservative raw limit.
            if len(value) <= 190 * 1024:
                return value
            image = image.resize((max(1, image.width // 2), max(1, image.height // 2)))


def media_steps(cache, metadata, audio, args):
    """A frame at end is reserved for the NEXT step; audio uses [start, end)."""
    import numpy as np
    duration = float(metadata['duration'])
    stop = min(duration, args.max_seconds) if args.max_seconds else duration
    frame_index = 0
    for start in range(math.ceil(stop)):
        end = min(start + 1., stop)
        frames = []
        while frame_index < len(metadata['frame_times']) and metadata['frame_times'][frame_index] < end:
            at = float(metadata['frame_times'][frame_index])
            if at >= start:
                frames.append((at, jpeg_frame(cache / metadata['frames'][frame_index], args)))
            frame_index += 1
        pcm = None
        padding = 0
        if metadata['has_audio']:
            segment = audio[round(start * 16000):round(end * 16000)]
            expected = round(end * 16000) - round(start * 16000)
            padding = max(0, expected - len(segment))
            if padding:
                # Some containers have a shorter audio track than video; silence
                # preserves alignment without reading a future audio interval.
                segment = np.pad(segment, (0, padding))
            pcm = (np.clip(segment, -1., 32767 / 32768) * 32768).astype('<i2').tobytes()
        yield {'start': float(start), 'end': end, 'frames': frames, 'pcm': pcm,
               'audio_padding_samples': padding}


def spoken_text(text):
    """Filter only an entire listen marker, allowing ASR case/punctuation changes."""
    if re.sub(r'[\W_]', '', text.casefold()) == 'silence':
        return ''
    return text.strip()


async def stream_sample(args, state, context, session_class):
    sample = context.sample
    cache, metadata, audio = media_cache(args.videos_dir / sample['video'], args.cache_dir)
    duration = float(metadata['duration'])
    stop = min(duration, args.max_seconds) if args.max_seconds else duration
    qtime = seconds(sample['question_time'])
    if not math.isfinite(qtime) or qtime < 0 or qtime >= stop:
        raise ValueError('question_time must precede the end of the observed input')
    system = context.build_prompt(PROTOCOL_PROMPT.read_text())
    inserted = False
    events = []
    with context.open_raw('.realtime.jsonl') as raw:
        def log(direction, payload):
            raw.write(json.dumps({'direction': direction,
                                 'wall_seconds': time.perf_counter() - context.started,
                                 **payload}, ensure_ascii=False) + '\n')
            raw.flush()

        session = session_class(args, state, log)
        try:
            await asyncio.wait_for(session.open(system), args.api_timeout)
            next_input = time.perf_counter()
            for step in media_steps(cache, metadata, audio, args):
                # Live video endpoints accept at most 1 FPS; never burst the prefix.
                await asyncio.sleep(max(0., next_input - time.perf_counter()))
                next_input = time.perf_counter() + 1.
                question = None
                if not inserted and step['start'] >= qtime:
                    question = user_request(sample)
                    inserted = True
                log('input', {'type': 'media_step', 'start_seconds': step['start'],
                    'end_seconds': step['end'], 'frame_times': [at for at, _ in step['frames']],
                    'has_source_audio': metadata['has_audio'],
                    'audio_padding_samples': step.get('audio_padding_samples', 0),
                    'question_inserted': question is not None})
                started = time.perf_counter()
                result = await asyncio.wait_for(session.step(step, question, inserted), args.api_timeout)
                if result is None:
                    continue
                first = result.pop('first_output_monotonic', None)
                result['first_output_wall_seconds'] = first - context.started if first is not None else None
                text = spoken_text(result['text'])
                log('output', {'type': 'step_result', 'end_seconds': step['end'],
                    'is_listen': not bool(text), 'step_wall_seconds': time.perf_counter() - started,
                    **result})
                if text:
                    events.append({'onset_seconds': step['end'], 'text': text,
                        'input_start_seconds': step['start'], 'input_end_seconds': step['end'],
                        'first_output_wall_seconds': result.get('first_output_wall_seconds'),
                        'step_wall_seconds': time.perf_counter() - started})
            if not inserted:
                raise ValueError('No media interval starts at or after question_time')
        finally:
            await session.close()
    return SampleResult(events=events, observed_seconds=stop,
                        status='partial' if stop < duration else 'complete')


def run_sample(args, state, context, session_class):
    context.started = time.perf_counter()
    return asyncio.run(stream_sample(args, state, context, session_class))
