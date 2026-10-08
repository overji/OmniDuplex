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


def run_config(a, base):
    config = {
        **MODEL_SETTINGS['livestar'],
        'model_path': str(a.model_path),
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
    }
    return config


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
    native_prompt = ("You are observing an incoming video stream. Before the user's request, "
        "observe silently. After the request, respond only when the SVeD policy activates. "
        "Available tools and agent_call schema are in this system message; only describe "
        "delegation to available tools, and never claim an operation succeeded.")
    prompt = context.build_prompt(native_prompt)
    if not hasattr(model, "system_message"):
        raise RuntimeError("LiveStar remote model does not expose its native system_message")
    model.system_message = prompt
    original_chat = model.chat
    def chat_with_request(this, chat_tokenizer, pixel_values, question, *args, **kwargs):
        match = re.search(r"Frame-(\d+):", question)
        if match and int(match.group(1)) == math.ceil(qtime):
            question = question + "\n" + user_request(sample)
        return original_chat(chat_tokenizer, pixel_values, question, *args, **kwargs)
    model.chat = MethodType(chat_with_request, model)

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
    with torch.inference_mode():
        summary = session.run(len(frames_px))
    model.chat = original_chat

    raw = [{"video_time_seconds": int(t), "text": text} for t, text in session.narration]
    events = [{"onset_seconds": int(t), "text": text} for t, text in session.narration
              if int(t) >= qtime and text.strip()]
    with context.open_raw(".json") as f:
        json.dump({"sample_index": index, "summary": summary, "narration": raw}, f,
                  ensure_ascii=False, indent=2)
    return SampleResult(events=events, observed_seconds=meta["duration"])
