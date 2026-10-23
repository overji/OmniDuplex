"""Run MiniCPM-o native full-duplex inference on the current timing benchmark."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
import fcntl
from pathlib import Path
from types import SimpleNamespace

from benchmark.data.timestamps import seconds
from benchmark.config.models import MODEL_SETTINGS
from benchmark.data.prompts import user_request
from benchmark.config.paths import MANIFESTS
from benchmark.data.media import media_cache
from benchmark.data.types import SampleResult


STRICT_CONFIG = True
SELECT_BEFORE_RESUME = True


def run_config(args, prompt):
    import torch
    import transformers

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        **MODEL_SETTINGS['minicpmo'],
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    manifest = MANIFESTS / "model_manifest.json"
    if manifest.exists():
        config["model_revision"] = json.loads(manifest.read_text())["revision"]
    return config


def load(args):
    import torch
    from transformers import AutoModel

    torch.set_num_threads(args.cpu_threads)
    print(f"Loading {args.model_path} on {args.device}", flush=True)
    # Transformers treats dots in local model folder names as Python module separators.
    model_target = args.model_path.resolve()
    if (model_target / 'full_training_model.json').is_file():
        suffix = hashlib.sha256(str(model_target).encode()).hexdigest()[:12]
        model_alias = args.cache_dir / 'full_training_models' / ('minicpmo_' + suffix)
        model_alias.parent.mkdir(parents=True, exist_ok=True)
    else:
        model_alias = args.cache_dir / "MiniCPM_o_4_5"
    with (args.cache_dir / "model_alias.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if model_alias.is_symlink():
            if model_alias.resolve() != model_target:
                model_alias.unlink()
                model_alias.symlink_to(model_target, target_is_directory=True)
        elif model_alias.exists():
            raise RuntimeError(f"Model cache alias path already exists: {model_alias}")
        else:
            model_alias.symlink_to(model_target, target_is_directory=True)
    model = AutoModel.from_pretrained(
        str(model_alias), trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        **({'init_tts': False} if (args.model_path / 'full_training_model.json').is_file() else {}),
    ).eval().to(args.device)
    if (args.model_path / 'full_training_model.json').is_file():
        from transformers import AutoTokenizer
        from benchmark.data.agent_token import full_model_identity
        full_model_identity(args.model_path)
        model.prepare_processor(tokenizer=AutoTokenizer.from_pretrained(str(model_alias), trust_remote_code=True))
        model.init_tts = lambda **kwargs: None
    if getattr(args, 'lora_adapter', None):
        from train.src.token_rows import load_adapter
        load_adapter(model, model_alias, args.lora_adapter)
    if args.unused_tts_device == "cpu":
        # Text-only runs never invoke TTS. Move its loaded weights off-device and
        # skip Token2wav initialization, which otherwise transiently allocates
        # several additional GiB on the GPU before being moved back to CPU.
        if getattr(model, 'tts', None) is not None:
            model.tts.to("cpu")
        model.init_tts = lambda *unused_args, **unused_kwargs: None
    model = model.as_duplex(generate_audio=MODEL_SETTINGS["minicpmo"]["generate_audio"], sliding_window_mode=args.sliding_window_mode,
        basic_window_high_tokens=args.window_high_tokens, basic_window_low_tokens=args.window_low_tokens)
    if args.unused_tts_device == "cpu":
        # Keep the unused TTS modules on CPU for the remainder of the run.
        if getattr(model.model, 'tts', None) is not None:
            model.model.tts.to("cpu")
        torch.cuda.empty_cache()
    print("Native full-duplex model ready", flush=True)

    return SimpleNamespace(model=model)


def run_sample(args, state, context):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed

    sample_index, sample = context.index, context.sample
    model = state.model
    with torch.inference_mode():
        set_seed(args.seed + sample_index)
        started = time.perf_counter()
        context.started = started
        target, metadata, audio = media_cache(args.videos_dir / sample["video"], args.cache_dir)
        duration = metadata["duration"]
        stop = min(duration, args.max_seconds) if args.max_seconds else duration
        question_time = seconds(sample["question_time"])
        model.force_listen_count = int(question_time)
        sample_prompt = context.build_prompt()
        model.prepare(prefix_system_prompt=sample_prompt)
        events = []
        token_stream = None
        if getattr(args, 'agent_output_protocol', 'text') == 'agent-token':
            from benchmark.data.agent_token import AgentStream
            token_stream = AgentStream(model.tokenizer)
        active_event = None
        question_sent = False
        print(f"START sample={sample_index} task={sample['task']} duration={stop:.3f}s question={question_time}s", flush=True)
        with context.open_raw(".chunks.jsonl") as raw:
            def record_result(result, clock, kind, prefill_seconds):
                nonlocal active_event
                entry = {"video_time_seconds": clock, "kind": kind,
                         "prefill_wall_seconds": prefill_seconds,
                         **{key: value for key, value in result.items() if key != "audio_waveform"}}
                raw.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                raw.flush()
                if token_stream is not None:
                    if question_sent:
                        event = token_stream.feed(result['generated_ids'], clock,
                            ended=result['end_of_turn'], interrupted=result['is_listen'])
                        if event is not None:
                            events.append(event)
                    return
                if (result.get('is_tool_call') or result.get('tool_calls')) and question_sent:
                    from benchmark.evaluation.agent_calls import native_event
                    events.append(native_event(result, clock, active_event))
                    active_event = None
                    return
                text = result.get("text", "")
                if text and not result["is_listen"] and question_sent:
                    if active_event is None:
                        active_event = {"onset_seconds": clock, "text": ""}
                    active_event["text"] += text
                    active_event["last_text_seconds"] = clock
                    if (getattr(args, 'agent_output_protocol', 'text') == 'structured'
                            and sample.get('mode') == 'agent' and '"tool_calls"' in active_event['text']):
                        active_event.setdefault('agent_call_onset_seconds', clock)
                if active_event is not None and (result["end_of_turn"] or result["is_listen"]):
                    if 'agent_call_onset_seconds' in active_event:
                        active_event['agent_call_complete_seconds'] = clock
                    events.append(active_event)
                    print(f"RESPONSE sample={sample_index} at={active_event['onset_seconds']:.3f}s {active_event['text'][:180]}", flush=True)
                    active_event = None

            for chunk_index in range(math.ceil(stop)):
                start = float(chunk_index)
                end = min(start + 1, stop)
                if not question_sent and start >= question_time:
                    question_sent = True
                    prefill = model.streaming_prefill(text_list=[user_request(sample)])
                    if not prefill["success"]:
                        raise RuntimeError(prefill)
                    # Text alone must not trigger generation. The current second's
                    # audiovisual prefill below always precedes the next output.
                frame_index = bisect.bisect_left(metadata["frame_times"], end) - 1
                frame_list = []
                if frame_index >= 0:
                    with Image.open(target / metadata["frames"][frame_index]) as image:
                        frame_list = [image.convert("RGB")]
                chunk = audio[int(start * MODEL_SETTINGS["minicpmo"]["audio_sample_rate"]):int(end * MODEL_SETTINGS["minicpmo"]["audio_sample_rate"])]
                if len(chunk) < MODEL_SETTINGS["minicpmo"]["audio_sample_rate"]:
                    chunk = np.pad(chunk, (0, MODEL_SETTINGS["minicpmo"]["audio_sample_rate"] - len(chunk)))
                then = time.perf_counter()
                prefill = model.streaming_prefill(audio_waveform=chunk, frame_list=frame_list,
                    max_slice_nums=1, batch_vision_feed=False)
                if not prefill["success"]:
                    raise RuntimeError(prefill)
                prefill_seconds = time.perf_counter() - then
                before = len(model.total_ids) if token_stream is not None else None
                result = model.streaming_generate(decode_mode=args.decode_mode,
                    max_new_speak_tokens_per_chunk=args.max_speak_tokens)
                if token_stream is not None:
                    result['generated_ids'] = list(model.total_ids[before:])
                record_result(result, end, "video_audio", prefill_seconds)
                if (chunk_index + 1) % 30 == 0:
                    print(f"PROGRESS sample={sample_index} video={end:.0f}/{stop:.0f}s wall={time.perf_counter()-started:.1f}s", flush=True)
        if token_stream is not None:
            pending = token_stream.finish(complete=False)
            if pending is not None:
                events.append(pending)
        if active_event is not None:
            events.append(active_event)
        return SampleResult(events=events, observed_seconds=stop,
                            status="partial" if stop < duration else "complete",
                            include_sample_type=False)
