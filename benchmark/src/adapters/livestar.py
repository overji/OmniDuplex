"""Run the official LiveStar base SVeD streaming session on timing samples."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace, MethodType

from benchmark.config.models import MODEL_SETTINGS, NATIVE_SETTINGS
from benchmark.data.prompts import user_request
from benchmark.data.media import media_cache
from benchmark.data.types import SampleResult
from benchmark.data.timestamps import seconds

NATIVE_PROMPT = ("You are observing an incoming video stream. Before the user's request, "
    "observe silently. After the request, respond only when the SVeD policy activates. "
    "Available tools and agent_call schema are in this system message; only describe "
    "delegation to available tools, and never claim an operation succeeded.")


def reuse_predictions(a, samples, assignment, config, base_prompt):
    """Import only verified non-Agent rows; never relabel an old Agent generation."""
    from benchmark.data.files import read_jsonl
    from benchmark.data.prompts import build_system_prompt
    source = a.reuse_non_agent.resolve()
    plan_path = source / 'parallel_plan.json'
    plan = json.loads(plan_path.read_text())
    previous = plan['arguments']
    if (plan['model'] != 'livestar' or plan['data_sha256'] != config['data_sha256']
            or plan['prompt_sha256'] != hashlib.sha256(base_prompt.encode()).hexdigest()
            or previous.get('agent_call_schema') != getattr(a, 'agent_call_schema', 'legacy')
            or any(previous.get(key) != str(getattr(a, key)) for key in ('model_path', 'videos_dir', 'official_code'))):
        raise ValueError('Incompatible LiveStar source inputs/model/prompt')
    allowed = set(assignment) if assignment is not None else set(range(len(samples)))
    result, seen = [], set()
    for worker in sorted(source.glob('worker_*')):
        old = json.loads((worker / 'run_config.json').read_text())
        expected = {key: value for key, value in config.items() if key != 'agent_input_protocol'}
        if any(old.get(key) != value for key, value in expected.items()):
            raise ValueError('LiveStar native model/sampling configuration changed')
        for record in read_jsonl(worker / 'predictions.jsonl', complete_only=True):
            index = record['sample_index']
            if index not in allowed or samples[index]['mode'] == 'agent' or record.get('status') != 'complete':
                continue
            if index in seen:
                raise ValueError('Duplicate reusable LiveStar sample')
            sample = samples[index]
            prompt = build_system_prompt(base_prompt, sample, NATIVE_PROMPT)
            if (any(record.get(key) != sample[key] for key in ('task', 'video', 'question'))
                    or record.get('system_prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()):
                raise ValueError('Reusable LiveStar record identity/prompt changed')
            seen.add(index)
            result.append({**record, 'reuse_source': str(worker / 'predictions.jsonl'),
                           'reuse_source_plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                           'reuse_scope': 'non_agent_unchanged_prompt_and_native_settings'})
    return result


def run_config(a, base):
    config = {
        **MODEL_SETTINGS['livestar'],
        'model_path': str(a.model_path),
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
    }
    return config


def narration_events(narration, question_time, duration, *, task_text=False, completions=None):
    """Native frame t is observable only after input through t+1 seconds."""
    from benchmark.data.task_text import event
    result = []
    for time_index, text in narration:
        if int(time_index) < question_time or not text.strip():
            continue
        clock = min(float(time_index) + 1, duration)
        complete = (completions or {}).get(int(time_index))
        result.append(event(text, clock, complete=complete, separator_time=clock, end_time=clock)
                      if task_text else {'onset_seconds': clock, 'text': text})
    return result


def load(a):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(a.official_code / "inference"))
    sys.path.insert(0, str(a.official_code))
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    import streaming_infer as native

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(str(a.model_path), trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_path), trust_remote_code=True, use_fast=False)

    gen_cfg = dict(**NATIVE_SETTINGS["livestar"]["generation"],
        max_new_tokens=MODEL_SETTINGS["livestar"]["max_new_tokens"])
    return SimpleNamespace(torch=torch, Image=Image, native=native, model=model, tokenizer=tokenizer, gen_cfg=gen_cfg)


def run_sample(a, state, context):
    index, sample = context.index, context.sample
    torch = state.torch
    Image = state.Image
    native = state.native
    model = state.model
    tokenizer = state.tokenizer
    gen_cfg = state.gen_cfg
    started = time.perf_counter()
    context.started = started
    cache, meta, _ = media_cache(a.videos_dir / sample["video"], a.cache_dir)
    qtime = seconds(sample["question_time"])
    prompt = context.build_prompt(NATIVE_PROMPT)
    if not hasattr(model, "system_message"):
        raise RuntimeError("LiveStar remote model does not expose its native system_message")
    model.system_message = prompt
    original_chat = model.chat
    task_text = getattr(a, 'agent_output_protocol', 'text') == 'task-text' and sample.get('mode') == 'agent'
    original_generate = getattr(model, 'generate', None)
    if task_text and original_generate is None:
        raise RuntimeError('LiveStar task-text requires native generation IDs for stop tracking')
    completions = {}
    captured = {}
    def generate_with_status(*args, **kwargs):
        from benchmark.probe.output import decoded_output
        output = original_generate(*args, **kwargs)
        captured['completion'] = decoded_output('', output[0], 0, tokenizer,
            kwargs.get('max_new_tokens', gen_cfg['max_new_tokens']), kwargs.get('eos_token_id')).generation_complete
        return output
    def chat_with_request(this, chat_tokenizer, pixel_values, question, *args, **kwargs):
        match = re.search(r"Frame-(\d+):", question)
        if match and int(match.group(1)) == math.ceil(qtime):
            question = question + "\n" + user_request(sample)
        captured.clear()
        result = original_chat(chat_tokenizer, pixel_values, question, *args, **kwargs)
        if match and 'completion' in captured:
            completions[int(match.group(1))] = captured['completion']
        return result
    model.chat = MethodType(chat_with_request, model)
    if task_text:
        model.generate = generate_with_status

    transform = native.build_transform(input_size=448)
    frames_px = []
    for relative in meta["frames"]:
        with Image.open(cache / relative) as image:
            tiles = native.dynamic_preprocess(image.convert("RGB"), image_size=448,
                                               use_thumbnail=True, max_num=MODEL_SETTINGS["livestar"]["max_num"])
        frames_px.append(torch.stack([transform(tile) for tile in tiles]))

    args = SimpleNamespace(**NATIVE_SETTINGS["livestar"]["session"],
        num_runs=MODEL_SETTINGS["livestar"]["num_runs"],
        alpha=MODEL_SETTINGS["livestar"]["alpha"])
    session = native.StreamingSession(model, tokenizer, gen_cfg, frames_px, args)
    try:
        with torch.inference_mode():
            summary = session.run(len(frames_px))
    finally:
        model.chat = original_chat
        if task_text:
            model.generate = original_generate

    def output_clock(t):
        return min(float(t) + 1, meta["duration"])
    raw = [{"video_time_seconds": output_clock(t), "native_frame_time_seconds": int(t),
            "text": text} for t, text in session.narration]
    events = narration_events(session.narration, qtime, meta['duration'],
        task_text=task_text,
        completions=completions)
    with context.open_raw(".json") as f:
        json.dump({"sample_index": index, "summary": summary, "narration": raw}, f,
                  ensure_ascii=False, indent=2)
    return SampleResult(events=events, observed_seconds=meta["duration"])
