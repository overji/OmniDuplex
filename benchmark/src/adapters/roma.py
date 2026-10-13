"""Run ROMA's native SpeakHead and gated response generation on timing samples."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from types import SimpleNamespace

from benchmark.config.models import MODEL_SETTINGS, NATIVE_SETTINGS
from benchmark.data.prompts import user_request
from benchmark.data.agent_inputs import AgentInputs
from benchmark.config.paths import CACHE
from benchmark.data.media import media_cache
from benchmark.data.types import SampleResult
from benchmark.data.timestamps import seconds


def run_config(a, base):
    config = {
        **MODEL_SETTINGS['roma'],
        'model_path': str(a.model_path),
        'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
    }
    return config


def load(a):
    sys.path.insert(0, str(a.roma_code))
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5OmniModel
    from src.llamafactory.model.loader import patch_processor
    from src.llamafactory.data.template import get_template_and_fix_tokenizer

    model = Qwen2_5OmniModel.from_pretrained(str(a.model_path), torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="flash_attention_2",
        trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_path), trust_remote_code=True,
        use_fast=True, padding_side="left", split_special_tokens=False)
    processor = AutoProcessor.from_pretrained(str(a.model_path), trust_remote_code=True)
    processor_args = SimpleNamespace(**NATIVE_SETTINGS["roma"]["processor"],
        video_fps=MODEL_SETTINGS["roma"]["video_fps"],
        audio_sampling_rate=MODEL_SETTINGS["roma"]["audio_sample_rate"],
        use_audio_in_video=MODEL_SETTINGS["roma"]["use_audio_in_video"])
    patch_processor(processor, tokenizer, processor_args)
    template = get_template_and_fix_tokenizer(tokenizer,
        SimpleNamespace(template="streaming_mix", train_on_prompt=False, tool_format=None))
    return SimpleNamespace(torch=torch, model=model, tokenizer=tokenizer, processor=processor, template=template)


def run_sample(a, state, context):
    index, sample = context.index, context.sample
    torch = state.torch
    model = state.model
    tokenizer = state.tokenizer
    processor = state.processor
    template = state.template
    assistant_prefix = [151645, 198, 151644, 77091, 198]
    user_prefix = [198, 151643, 872, 198]
    started = time.perf_counter()
    context.started = started
    video_path = a.videos_dir / sample["video"]
    cache, meta, _ = media_cache(video_path, CACHE)
    qtime = seconds(sample["question_time"])
    native = ("You are a real-time audiovisual assistant. Observe only the stream already "
        "received. Before the user request arrives, remain silent. After it arrives, use the "
        "native ROMA SpeakHead to decide when to respond. Available tools and agent_call "
        "schema are in the system message. Never claim a tool ran successfully.")
    system = context.build_prompt(native)

    duration = meta["duration"]
    audio_path = str(cache / "audio.wav")
    prompt = [{"text": "<video>\nObserve the incoming audiovisual stream chronologically.",
               "time": 0.0, "duration": duration, "audio": audio_path}]
    videos = [[str(video_path)]]
    responses = [{"text": "", "time": 0.0, "audio": None}]
    messages = template.mm_plugin.process_messages([[prompt, responses]], [], videos, [], processor, mode="infer")
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, "")
    chunks = [row[0] for row in encoded_pairs]
    mm_inputs = template.mm_plugin.get_mm_inputs([], [videos[0]], [], [0], [], [1],
        [chunks[0]], processor, messages=[[prompt, responses]])

    input_ids = []
    video_tokens = 0
    past_key_values = None
    rope_delta = None
    events, raw = [], []
    qframe = math.ceil(qtime)
    agent_inputs = AgentInputs(sample)
    for i, source_chunk in enumerate(chunks):
        chunk = list(source_chunk)
        if i == qframe:
            chunk.extend(tokenizer.encode(user_request(sample), add_special_tokens=False))
        for text in agent_inputs.take(float(i)):
            chunk.extend(tokenizer.encode(text, add_special_tokens=False))
        input_ids.extend(chunk)
        old_video_tokens = video_tokens
        video_tokens += chunk.count(151656)
        features = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=model.device),
            "attention_mask": torch.ones([1, len(input_ids)], dtype=torch.long, device=model.device),
            "video_grid_thw": mm_inputs["video_grid_thw"].clone().to(model.device),
            "pixel_values_videos": mm_inputs["pixel_values_videos"][old_video_tokens * 4:video_tokens * 4].to(model.dtype).to(model.device),
            "input_features": mm_inputs["input_features"][:, :, i * 100:min((i + 1) * 100, mm_inputs["input_features"].shape[2])].to(model.dtype).to(model.device),
            "feature_attention_mask": mm_inputs["feature_attention_mask"][:, i * 100:min((i + 1) * 100, mm_inputs["feature_attention_mask"].shape[1])].to(model.device),
            "video_second_per_grid": mm_inputs["video_second_per_grid"].to(model.dtype).to(model.device),
        }
        features["video_grid_thw"][0, 0] = 1
        audio_lens = torch.sum(features["feature_attention_mask"], dim=1)
        position_ids, current_rope = model.thinker.get_interleaved_rope_index(
            features["input_ids"][:, -len(chunk):], None, features["video_grid_thw"],
            features["attention_mask"][:, -len(chunk):], use_audio_in_video=MODEL_SETTINGS["roma"]["use_audio_in_video"],
            audio_seqlens=audio_lens)
        final_rope = current_rope
        cache_position = torch.arange(0, len(input_ids), dtype=torch.long, device=model.device)[-len(chunk):]
        if rope_delta is not None:
            position_ids += cache_position[0] + rope_delta
            final_rope += rope_delta
        probe = {**features, "input_ids": features["input_ids"][:, -len(chunk):],
            "use_cache": True, "output_hidden_states": True, "return_dict": True,
            "past_key_values": past_key_values, "rope_deltas": final_rope,
            "position_ids": position_ids, "cache_position": cache_position}
        with torch.inference_mode():
            out = model.thinker(**probe)
        hs = out.hidden_states[1] if isinstance(out.hidden_states, tuple) and isinstance(out.hidden_states[1], (list, tuple)) else out.hidden_states
        bsz, seq_len, hidden = hs[-1].shape
        anchor = torch.full((bsz, 1, hidden), seq_len - 1, dtype=torch.long, device=model.device)
        mixed = 0.0
        layers = getattr(model.thinker, "gate_layer_ids", [-4, -3, -2, -1])
        for weight, layer in zip(model.thinker.gate_mixer.weights(), layers):
            layer = layer if layer >= 0 else len(hs) + layer
            mixed = mixed + weight * torch.gather(hs[layer], 1, anchor)
        logit = model.thinker.gate_head_pro_fc2(model.thinker.gate_head_pro_act(
            model.thinker.gate_head_pro_fc1(mixed))).squeeze()
        probability = torch.sigmoid(logit).item()
        past_key_values, rope_delta = out.past_key_values, out["rope_deltas"]
        event = {"video_time_seconds": float(i + 1), "speak_probability": probability,
                 "threshold": MODEL_SETTINGS["roma"]["threshold"], "triggered": bool(i >= qframe and probability > MODEL_SETTINGS["roma"]["threshold"])}
        if i >= qframe and probability > MODEL_SETTINGS["roma"]["threshold"]:
            input_ids.extend(assistant_prefix)
            features["input_ids"] = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            features["attention_mask"] = torch.ones([1, len(input_ids)], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                generated = model.generate(**features, thinker_max_new_tokens=MODEL_SETTINGS["roma"]["thinker_max_new_tokens"],
                    use_audio_in_video=MODEL_SETTINGS["roma"]["use_audio_in_video"], return_audio=False, streaming=True,
                    past_key_values=past_key_values, output_scores=True, mode="infer",
                    rope_deltas=rope_delta)
            ids = generated.sequences[0, len(input_ids):]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            if text:
                event["text"] = text
                events.append({"onset_seconds": float(i + 1), "text": text})
            input_ids.extend(ids.tolist())
            past_key_values, rope_delta = generated.past_key_values, generated["rope_deltas"]
            input_ids.extend(user_prefix)
        raw.append(event)
    with context.open_raw(".json") as f:
        json.dump({"sample_index": index, "decisions": raw,
                   "agent_inputs": agent_inputs.delivered}, f, ensure_ascii=False, indent=2)
    return SampleResult(events=events, observed_seconds=duration)
