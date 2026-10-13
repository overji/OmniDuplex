"""Model-independent sample lifecycle and artifact persistence."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.data.agent_inputs import AGENT_INPUT_PROTOCOL, agent_results
from benchmark.data.prompts import build_system_prompt
from benchmark.data.types import SampleResult
from benchmark.evaluation.timing import write_metrics


@dataclass
class SampleContext:
    index: int
    sample: dict
    base_prompt: str
    output_dir: Path
    started: float = 0.0
    system_prompt: str = ""

    def build_prompt(self, native_protocol=""):
        self.system_prompt = build_system_prompt(self.base_prompt, self.sample, native_protocol)
        append_jsonl(self.output_dir / "sample_system_prompts.jsonl", {
            "sample_index": self.index, "system_prompt": self.system_prompt,
        })
        return self.system_prompt

    def open_raw(self, suffix):
        return (self.output_dir / f"sample_{self.index:03d}{suffix}").open("w")


def select_samples(samples, completed, args, *, before_resume=False):
    if before_resume:
        if args.sample_start < 0 or args.sample_step < 1:
            raise ValueError("sample-start must be nonnegative and sample-step must be positive")
        selected = list(enumerate(samples))[args.sample_start::args.sample_step]
        selected = selected[:args.max_samples or len(selected)]
        return [(index, sample) for index, sample in selected if index not in completed]
    selected = [(index, sample) for index, sample in enumerate(samples) if index not in completed]
    return selected[:args.max_samples] if args.max_samples else selected


def run(args, adapter, *, assignment=None):
    samples = json.loads(args.data.read_text())
    base_prompt = args.system_prompt.read_text()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "cache_dir"):
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = adapter.run_config(args, base_prompt)
    config["agent_input_protocol"] = AGENT_INPUT_PROTOCOL
    config_path = args.output_dir / "run_config.json"
    strict = getattr(adapter, "STRICT_CONFIG", False)
    if (args.output_dir / "predictions.jsonl").exists():
        previous = json.loads(config_path.read_text()) if config_path.exists() else {}
        if previous.get("agent_input_protocol") != AGENT_INPUT_PROTOCOL:
            raise ValueError("Agent input protocol changed; use a new output directory")
    if strict and config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; use a new output directory")
    write_json(config_path, config)
    prediction_path = args.output_dir / "predictions.jsonl"
    if assignment is not None:
        recover_jsonl_tail(prediction_path)
    predictions = read_jsonl(prediction_path)
    completed = {prediction["sample_index"] for prediction in predictions}
    if assignment is None:
        selected = select_samples(samples, completed, args,
                                  before_resume=getattr(adapter, "SELECT_BEFORE_RESUME", False))
    else:
        selected = [(index, samples[index]) for index in assignment if index not in completed]
    if not selected:
        if strict:
            write_metrics(samples, predictions, args.output_dir / "metrics.json")
        return
    state = adapter.load(args)
    for index, sample in selected:
        # References are reserved for evaluation, never the model-facing context.
        model_sample = {key: sample[key] for key in (
            "id", "task", "video", "question", "question_time", "available_tools",
            "frontend_context", "sample_type", "mode", "ability",
        ) if key in sample}
        model_sample["agent_results"] = agent_results(sample)
        context = SampleContext(index, model_sample, base_prompt, args.output_dir,
                                started=time.perf_counter())
        result: SampleResult = adapter.run_sample(args, state, context)
        prediction = {
            "sample_index": index, "task": sample["task"], "video": sample["video"],
            "question": sample["question"],
        }
        if result.include_sample_type:
            prediction["sample_type"] = sample.get("sample_type")
        prediction.update({
            "system_prompt_sha256": hashlib.sha256(context.system_prompt.encode()).hexdigest(),
            "status": result.status, "observed_seconds": result.observed_seconds,
            "wall_seconds": time.perf_counter() - context.started, "events": result.events,
        })
        append_jsonl(prediction_path, prediction)
        predictions.append(prediction)
        metrics = write_metrics(samples, predictions, args.output_dir / "metrics.json")
        message = f"DONE sample={index} events={len(result.events)}"
        if strict:
            message += f" aggregate={metrics['groups']['overall']}"
        print(message, flush=True)
