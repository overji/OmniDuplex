"""Native MOSS realtime sessions, paced frames and consumption-based timestamps.

The official API returns text without timestamps. Instrument queue consumption
and the completed model forward, rather than stamping text with the producer's
latest queued frame. This preserves asynchronous native decoding without giving
queued, unprocessed frames credit in timing evaluation.
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager
import hashlib
import json
import math
import queue
import re
import signal
import threading
import time
from types import SimpleNamespace

from benchmark.config.models import MODEL_SETTINGS
from benchmark.data.media import media_cache
from benchmark.data.prompts import user_request
from benchmark.data.timestamps import seconds
from benchmark.data.types import SampleResult

STRICT_CONFIG = True
RESTART_EXIT_CODE = 75
NATIVE_PROMPT = (
    'You observe a continuous video stream. Before the user question, stay silent. '
    'After the question, respond only when observed evidence warrants a reply; '
    'otherwise emit <|silence|> and continue observing. '
    'Preserve the native realtime response and silence control tokens.'
)
BOUNDARIES = re.compile(r'<\|(?:silence|response|round_start|im_end|assistant)\|>|<go>')


def run_config(args, prompt):
    for name in ('sample_timeout_seconds', 'drain_seconds'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if args.max_tokens_per_second < 1 or args.max_new_tokens < 1:
        raise ValueError('Token limits must be positive')
    return {**MODEL_SETTINGS['moss_vl'], 'model_path': str(args.model_path),
            'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            **{key: getattr(args, key) for key in ('sample_timeout_seconds', 'drain_seconds',
                'max_tokens_per_second', 'max_new_tokens', 'attention_backend')}}


def load(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True,
        local_files_only=True, frame_extract_num_threads=1)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation=args.attention_backend).eval()
    return SimpleNamespace(model=model, processor=processor, Image=Image, restart_required=False)


class InputClock:
    """Only a finished forward makes drained media available to an output."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = self.consumed = None
        self.prompt_pending = self.prompt_consumed = False

    def forwarded(self, *unused):
        with self.lock:
            self.consumed = self.pending
            self.prompt_consumed = self.prompt_pending


class InputQueue(queue.Queue):
    def __init__(self, clock, *, prompt=False, maxsize=0):
        super().__init__(maxsize)
        self.clock, self.prompt = clock, prompt

    def _get(self):
        value = super()._get()
        with self.clock.lock:
            if self.prompt:
                self.clock.prompt_pending = True
            else:
                self.clock.pending = float(value[1])
        return value


class OutputQueue(queue.Queue):
    def __init__(self, clock, started):
        super().__init__()
        self.clock, self.started = clock, started

    def put(self, text, block=True, timeout=None):
        with self.clock.lock:
            item = dict(text=str(text), onset_seconds=self.clock.consumed,
                        question_consumed=self.clock.prompt_consumed,
                        wall_seconds=time.perf_counter() - self.started)
        return super().put(item, block=block, timeout=timeout)


def narration_events(chunks, *, task_text=False):
    """Merge native text fragments until silence/end; retain the first token clock."""
    from benchmark.data.task_text import event
    if not chunks:
        return []
    text = ''.join(c['text'] for c in chunks)
    starts, offset = [], 0
    for chunk in chunks:
        starts.append(offset)
        offset += len(chunk['text'])
    events, parts, piece_ends, first, last = [], [], [], None, None

    def flush(complete):
        nonlocal parts, piece_ends, first, last
        joined = ''.join(parts)
        body = joined.strip()
        if body and first is not None:
            marker = joined.find('<|task_start|>')
            separator_time = next((clock for end, clock in piece_ends
                if end >= marker + len('<|task_start|>')), None) if marker >= 0 else None
            events.append(event(body, first, complete=complete,
                                separator_time=separator_time, end_time=last)
                          if task_text else dict(text=body, onset_seconds=first))
        parts, piece_ends, first, last = [], [], None, None

    def add(start, stop):
        nonlocal first, last
        while start < stop:
            index = bisect.bisect_right(starts, start) - 1
            end = min(stop, starts[index + 1] if index + 1 < len(starts) else len(text))
            chunk = chunks[index]
            piece = text[start:end]
            if chunk['question_consumed'] and chunk['onset_seconds'] is not None:
                if first is None and piece.strip():
                    first = chunk['onset_seconds']
                parts.append(piece)
                piece_ends.append(((piece_ends[-1][0] if piece_ends else 0) + len(piece), chunk['onset_seconds']))
                last = chunk['onset_seconds']
            start = end

    position = 0
    for match in BOUNDARIES.finditer(text):
        add(position, match.start())
        # Native MOSS repeats <|response|> while continuing a speaking turn;
        # it is not a new response onset. Only silence/end/new-user boundaries
        # close the current utterance.
        if match.group() not in ('<|response|>', '<|assistant|>', '<go>'):
            flush(True)
        position = match.end()
    add(position, len(text))
    flush(False)
    return events


@contextmanager
def deadline(seconds_limit):
    """The main thread stays responsive while the native model owns its thread."""
    def expired(signum, frame):
        raise TimeoutError('MOSS sample wall-clock limit exceeded')
    previous = signal.signal(signal.SIGALRM, expired)
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        signal.signal(signal.SIGALRM, previous)
        raise RuntimeError('Cannot nest MOSS sample deadline inside another alarm')
    signal.setitimer(signal.ITIMER_REAL, seconds_limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_sample(args, state, context):
    started = time.perf_counter()
    context.started = started
    clock, chunks, session, hook = InputClock(), [], None, None
    status, error = 'complete', None
    task_text = getattr(args, 'agent_output_protocol', 'text') == 'task-text' and context.sample.get('mode') == 'agent'
    try:
        with deadline(args.sample_timeout_seconds):
            cache, meta, _ = media_cache(args.videos_dir / context.sample['video'], args.cache_dir)
            prompt = context.build_prompt(NATIVE_PROMPT)
            question_time = seconds(context.sample['question_time'])
            session = state.model.create_realtime_session(state.processor,
                initial_prompt='', system_prompt=prompt,
                frame_queue_size=MODEL_SETTINGS['moss_vl']['frame_queue_size'],
                max_tokens_per_turn=args.max_tokens_per_second,
                max_new_tokens=args.max_new_tokens, do_sample=False)
            session._frame_queue = InputQueue(clock, maxsize=MODEL_SETTINGS['moss_vl']['frame_queue_size'])
            session._prompt_queue = InputQueue(clock, prompt=True)
            output = OutputQueue(clock, started)
            session._output_queue = output
            hook = state.model.register_forward_hook(clock.forwarded)
            session.start()

            def drain():
                while True:
                    try:
                        item = output.get_nowait()
                    except queue.Empty:
                        break
                    chunks.append(item)
                session._raise_if_failed()
                if not session.active:
                    raise RuntimeError('MOSS session ended before video replay completed')

            def until(wall):
                while time.perf_counter() < wall:
                    drain()
                    time.sleep(min(.02, max(0., wall - time.perf_counter())))

            replay_start, question_sent = time.perf_counter(), False
            for timestamp, relative in zip(meta['frame_times'], meta['frames']):
                end = min(float(timestamp) + 1., meta['duration'])
                until(replay_start + end)
                with state.Image.open(cache / relative) as image:
                    frame = image.convert('RGB').copy()
                # No producer-side queue eviction: every selected frame is evidence.
                if not question_sent and end >= question_time:
                    session.push_prompt_frame(user_request(context.sample), frame, timestamp=end, drop_oldest=False)
                    question_sent = True
                else:
                    session.push_frame(frame, timestamp=end, drop_oldest=False)
                drain()
            until(replay_start + meta['duration'] + args.drain_seconds)
            if not question_sent:
                raise ValueError('Question time exceeds observed video')
            if (session.pending_frames or not clock.prompt_consumed
                    or clock.consumed is None or clock.consumed < end):
                status, error = 'partial', 'Input not fully consumed after the configured drain window'
    except TimeoutError as exc:
        status, error = 'timeout', str(exc)
    except Exception as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    finally:
        if session is not None:
            try:
                session.close(timeout=5.)
            except Exception as exc:
                state.restart_required = True
                status = 'timeout' if status == 'timeout' else 'error'
                error = (error or '') + f'; session close failed: {exc}'
            while True:
                try:
                    chunks.append(session._output_queue.get_nowait())
                except queue.Empty:
                    break
        if hook is not None:
            hook.remove()
    with context.open_raw('.json') as stream:
        json.dump(dict(sample_index=context.index, status=status, error=error,
                       sample_timeout_seconds=args.sample_timeout_seconds, chunks=chunks),
                  stream, ensure_ascii=False, indent=2)
    print(f'MOSS sample={context.index} status={status} observed={clock.consumed} error={error}', flush=True)
    return SampleResult(events=narration_events(chunks, task_text=task_text),
                        observed_seconds=clock.consumed or 0., status=status)


def after_sample(args, state):
    """Persist the terminal row before restarting an unresponsive native session."""
    if state.restart_required:
        raise SystemExit(RESTART_EXIT_CODE)
