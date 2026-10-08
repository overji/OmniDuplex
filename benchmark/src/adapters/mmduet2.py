"""Run MMDuet2's official online inference client on all timing samples."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

from benchmark.config.models import MODEL_SETTINGS, NATIVE_SETTINGS
from benchmark.data.prompts import user_request
from benchmark.data.media import media_cache
from benchmark.data.types import SampleResult
from benchmark.data.timestamps import seconds


def run_config(a, base):
    config = {
        **MODEL_SETTINGS['mmduet2'],
        'model_path': str(a.model_path),
        'official_code': str(a.official_code),
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
    }
    return config


def load(a):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(a.official_code))
    spec = importlib.util.spec_from_file_location("mmduet_inference", a.official_code / "inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import torch
    from transformers import AutoProcessor

    model_cls = module.Qwen2_5_VLForConditionalGeneration
    model = model_cls.from_pretrained(str(a.model_path), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").eval().to(a.device)
    processor = AutoProcessor.from_pretrained(str(a.model_path))
    return SimpleNamespace(module=module, model=model, processor=processor)


def run_sample(a, state, context):
    index, sample = context.index, context.sample
    module = state.module
    model = state.model
    processor = state.processor
    import torch
    native_protocol = ('Use the official MMDuet2 streaming policy. Before the user request, '
        'observe silently. For an input that requires no reply, emit exactly NO REPLY. Do not '
        'invent tool execution or claim success. Available tools and agent_call schema are '
        'given in the system message; only delegate available capabilities.')
    started = time.perf_counter()
    context.started = started
    cache, meta, _ = media_cache(a.videos_dir / sample["video"], a.cache_dir)
    qtime = seconds(sample["question_time"])
    prompt = context.build_prompt(native_protocol)
    args = SimpleNamespace(**NATIVE_SETTINGS["mmduet2"]["inference"],
        system_prompt=prompt,
        do_sample=MODEL_SETTINGS["mmduet2"]["sampling"]["do_sample"])
    client = module.ProactiveInferenceClient(args=args, model=model, processor=processor)
    client.set_fps(frame_interval=float(MODEL_SETTINGS["mmduet2"]["frame_interval_seconds"]))
    conversation = [{"role": "system", "content": prompt}]
    frame_files = meta["frames"]
    from PIL import Image
    inserted_question = False
    for frame_time, relative in zip(meta["frame_times"], frame_files):
        if not inserted_question and frame_time >= qtime:
            conversation.append({"role": "user", "content": user_request(sample)})
            inserted_question = True
        image_path = cache / relative
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        turn = {"role": "user", "content": [{"type": "text", "text": f"Frame at {frame_time:.3f}s"},
                {"type": "image", "image": image}]}
        if frame_time < qtime:
            turn["skip_inference"] = True
        conversation.append(turn)
    if not inserted_question:
        conversation.append({"role": "user", "content": user_request(sample)})
    client.input_query_stream(conversation)
    with torch.inference_mode():
        result = client.inference()
    events = []
    raw = []
    question_replied = False
    for turn in result["conversation"]:
        if turn["role"] != "assistant":
            continue
        text = turn.get("content", "").strip()
        at = float(turn.get("time", 0))
        raw.append({"video_time_seconds": at, "text": text})
        if text and text.upper() != "NO REPLY" and at >= qtime:
            if not question_replied:
                at = max(at, qtime)
                question_replied = True
            events.append({"onset_seconds": at, "text": text})
    with context.open_raw(".json") as f:
        json.dump({"sample_index": index, "conversation": raw}, f, ensure_ascii=False, indent=2)
    return SampleResult(events=events, observed_seconds=meta["duration"])
