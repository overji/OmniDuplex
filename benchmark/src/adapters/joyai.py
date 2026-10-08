"""Stream causal frames through JoyAI's official realtime adapter endpoint."""
from __future__ import annotations

import hashlib
import json
import re
import time
from types import SimpleNamespace
from urllib.parse import quote

from benchmark.config.models import MODEL_SETTINGS
from benchmark.data.prompts import user_request
from benchmark.data.media import media_cache
from benchmark.data.types import SampleResult
from benchmark.data.timestamps import seconds




def run_config(a, base):
    config = {
        **MODEL_SETTINGS['joyai'],
        'model_path': str(a.model_path),
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
        'adapter_url': a.adapter_url,
    }
    return config


def load(a):
    import requests
    return SimpleNamespace(requests=requests)


def run_sample(a, state, context):
    index, sample = context.index, context.sample
    requests = state.requests
    native = ("Follow JoyAI's native interaction action protocol exactly: emit </silence> "
        "when no response is needed, or </response> followed by one concise response. "
        "Before the user's request arrives, remain silent. Use available tools and the "
        "agent_call schema in this system message; never claim unexecuted work succeeded.")
    started = time.perf_counter()
    context.started = started
    cache, meta, _ = media_cache(a.videos_dir / sample["video"], a.cache_dir)
    qtime = seconds(sample["question_time"])
    system = context.build_prompt(native)
    session = f"timing-benchmark-{index:03d}"
    requests.post(a.adapter_url.rsplit("/v1/", 1)[0] + "/v1/streaming/reset",
        json={}, headers={"x-streaming-session": session}, timeout=60).raise_for_status()
    raw, events = [], []
    inserted_query = False
    for frame_time, relative in zip(meta["frame_times"], meta["frames"]):
        frame_path = (cache / relative).resolve()
        content = []
        if not inserted_query and frame_time >= qtime:
            content.append({"type": "text", "text": user_request(sample)})
            inserted_query = True
        frame_uri = "file://" + quote(str(frame_path))
        content.append({"type": "image_url", "image_url": {"url": frame_uri}})
        payload = {"model": a.model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": content}],
            "frame_time_ranges": [f"{frame_time:.3f}s"]}
        response = requests.post(a.adapter_url, json=payload,
            headers={"x-streaming-session": session}, timeout=360)
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"].get("content", "")
        raw_text = body.get("streamingharness", {}).get("raw_content", message)
        at = float(frame_time)
        item = {"video_time_seconds": at, "content": message, "raw_content": raw_text,
                "timing": body.get("streamingharness", {}).get("timing")}
        raw.append(item)
        if at >= qtime:
            match = re.search(r"</response>\s*(.*?)(?:</|$)", raw_text, flags=re.S)
            if match and match.group(1).strip():
                events.append({"onset_seconds": at, "text": match.group(1).strip()})
    if not inserted_query:
        raise RuntimeError(f"question_time {qtime} exceeds last observed frame for sample {index}")
    with context.open_raw(".json") as f:
        json.dump({"sample_index": index, "turns": raw}, f, ensure_ascii=False, indent=2)
    return SampleResult(events=events, observed_seconds=meta["duration"])
