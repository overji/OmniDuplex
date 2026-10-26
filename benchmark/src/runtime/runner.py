"""Model-independent sample lifecycle and artifact persistence."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.data.agent_inputs import input_protocol
from benchmark.data.agent_schema import adapt_sample
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
    agent_output_protocol: str = "text"
    native_protocol: str = ""
    task_text_model: str = 'minicpmo'

    def build_prompt(self, native_protocol=""):
        native_protocol = native_protocol or self.native_protocol
        self.system_prompt = build_system_prompt(self.base_prompt, self.sample, native_protocol)
        if self.agent_output_protocol == 'task-text' and self.sample.get('mode') == 'agent':
            from benchmark.data.task_text import build_prompt
            self.system_prompt = build_prompt(self.sample, self.task_text_model)
        if self.agent_output_protocol == "structured" and self.sample.get("mode") == "agent":
            from benchmark.config.paths import CONFIGS
            name = 'agent_task_start_structured.txt' if self.sample.get('_agent_call_schema') == 'task-start' else 'agent_structured.txt'
            self.system_prompt += "\n\n" + (CONFIGS / 'prompts' / name).read_text()
        if self.agent_output_protocol == 'agent-token' and self.sample.get('mode') == 'agent':
            from benchmark.config.paths import CONFIGS
            self.system_prompt += '\n\n' + (CONFIGS / 'prompts/agent_token.txt').read_text()
        if self.sample.get('_agent_call_schema') == 'task-start':
            self.system_prompt = self.system_prompt.replace('agent_call', 'task_start')
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
    samples = [adapt_sample(s, getattr(args, 'agent_call_schema', 'legacy')) for s in json.loads(args.data.read_text())]
    base_prompt = args.system_prompt.read_text()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(args, "cache_dir"):
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    config = adapter.run_config(args, base_prompt)
    config["agent_input_protocol"] = input_protocol(args)
    config['agent_call_schema'] = getattr(args, 'agent_call_schema', 'legacy')
    config_path = args.output_dir / "run_config.json"
    strict = getattr(adapter, "STRICT_CONFIG", False)
    if (args.output_dir / "predictions.jsonl").exists():
        previous = json.loads(config_path.read_text()) if config_path.exists() else {}
        if previous.get("agent_input_protocol") != input_protocol(args):
            raise ValueError("Agent input protocol changed; use a new output directory")
    if strict and config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; use a new output directory")
    write_json(config_path, config)
    prediction_path = args.output_dir / "predictions.jsonl"
    if assignment is not None:
        recover_jsonl_tail(prediction_path)
    predictions = read_jsonl(prediction_path)
    if hasattr(adapter, 'reuse_predictions') and getattr(args, 'reuse_non_agent', None):
        reused = adapter.reuse_predictions(args, samples, assignment, config, base_prompt)
        completed_before = {p['sample_index'] for p in predictions}
        for prediction in reused:
            if prediction['sample_index'] not in completed_before:
                append_jsonl(prediction_path, prediction)
                predictions.append(prediction)
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
            "_agent_call_schema",
        ) if key in sample}
        context = SampleContext(index, model_sample, base_prompt, args.output_dir,
                                started=time.perf_counter(),
                                agent_output_protocol=getattr(args, "agent_output_protocol", "text"),
                                task_text_model=getattr(adapter, '__name__', 'minicpmo').rsplit('.', 1)[-1].replace('_ablation', ''))
        result: SampleResult = adapter.run_sample(args, state, context)
        if sample.get('mode') == 'agent' and (getattr(args, 'agent_output_protocol', 'text') == 'structured'
                or sample.get('_agent_call_schema') == 'task-start'):
            from benchmark.evaluation.agent_calls import parse_event
            from benchmark.data.agent_schema import validate_task_start
            for event in result.events:
                parsed = parse_event(event)
                if sample.get('_agent_call_schema') == 'task-start':
                    valid = bool(parsed['calls']) and bool(sample.get('available_tools')) and all(validate_task_start(c) for c in parsed['calls'])
                    event['agent_schema_valid'] = valid
                    if parsed['calls'] and not valid:
                        parsed['status'] = 'malformed'
                event["agent_parse_status"] = parsed["status"]
                event["response_text"] = parsed["response"]
                event["tool_calls"] = parsed["calls"]
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
