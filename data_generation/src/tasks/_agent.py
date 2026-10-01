"""Program-controlled capability contexts and triggered agent sample contracts."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..agent_tools import TASK_TOOLS, TOOLS, WRITE_TOOLS
from ..common import format_mmss, timestamp_seconds
from ..video_caption import load_caption, probe_video_duration
from . import _shared


@dataclass(frozen=True)
class GenerationContext(_shared.GenerationContext):
    sample_type: str = "positive"
    available_tools: tuple[str, ...] = ()
    required_missing_tool: str | None = None


GenerationRequest = _shared.GenerationRequest
PROMPT_DIR = _shared.PROMPT_ROOT / "agent"


def frontend_context(context: GenerationContext) -> dict[str, Any]:
    if context.sample_type == "no_context":
        return {"system_prompt": "You are an audiovisual assistant. State capability limitations honestly.", "tools": []}
    text = (PROMPT_DIR / "capabilities.md").read_text(encoding="utf-8")
    prefix, definition = text.split("Frontend function definition:\n", 1)
    available = "\n".join(f"- {name}: {TOOLS[name]}" for name in context.available_tools)
    return {
        "system_prompt": prefix.replace("{{AVAILABLE_TOOLS}}", available).strip(),
        "tools": [json.loads(definition)],
    }


def build_request(path: Path, video_path: Path, task: str) -> GenerationRequest:
    video = video_path.expanduser().resolve()
    duration = probe_video_duration(video)
    caption = load_caption(path, duration)
    sample_type = config.choose_agent_sample_type()
    catalog = TASK_TOOLS[task]
    missing = random.choice(catalog) if sample_type != "positive" else None
    available = () if sample_type == "no_context" else tuple(t for t in catalog if t != missing)
    context = GenerationContext(
        duration_seconds=duration,
        preferred_modality=config.choose_preferred_modality(),
        question_position=config.choose_question_position(duration),
        agent_delay_seconds=config.choose_agent_delay(duration) if sample_type == "positive" else None,
        initial_response_delay_seconds=config.choose_initial_response_delay(duration) if sample_type == "positive" else None,
        sample_type=sample_type,
        available_tools=available,
        required_missing_tool=missing,
    )
    prompt = (PROMPT_DIR / "generation.md").read_text(encoding="utf-8")
    prompt += "\n" + (PROMPT_DIR / f"{task}.md").read_text(encoding="utf-8")
    question_policy = (
        "question_time must be 00:00." if context.question_position == "start" else
        f"question_time must be after 00:00 and before {format_mmss(duration)}."
    )
    if sample_type == "positive":
        question_policy = _shared._question_policy(context, historical=False)
        timing = (
            "Return skip if no legal trigger and result fit in the video. "
            f"Backend delay: {context.agent_delay_seconds} seconds. "
            "If backend delay is None, return skip."
        )
    else:
        timing = (
            "Refuse immediately at question_time. No initial silent interval or backend delay applies. "
            f"Choose a task genuinely requiring {missing}: {TOOLS[missing]}. "
            "The missing tool is a generator constraint, NOT an available frontend capability. "
            "Do not emit skip instead of a capability refusal."
        )
    prompt += (
        f"\nDuration: {format_mmss(duration)}. Preferred modality: {context.preferred_modality}; "
        "fall back to the other modality only if unsuitable.\n"
        f"Sample type: {sample_type}. {question_policy}\n{timing}\n"
        "Tools belonging to this task family (generator reference, not frontend availability):\n"
        + "\n".join(f"- {name}: {TOOLS[name]}" for name in catalog)
    )
    user = {"caption": caption, "frontend_context": frontend_context(context)}
    return GenerationRequest(prompt, json.dumps(user, ensure_ascii=False, indent=2), context, video)


def _tool_list(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{label} must be a string array")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique tool names")
    return set(value)


def _validate_trigger(response: dict[str, Any], context: GenerationContext, task: str) -> None:
    trigger = response.get("trigger")
    if not isinstance(trigger, dict):
        raise ValueError("trigger must be an object")
    if trigger.get("kind") != task.removesuffix("_agent") or trigger.get("satisfied") is not True:
        raise ValueError("trigger kind or satisfaction is invalid")
    time = timestamp_seconds(response["trigger_time"])
    evidence = trigger.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("trigger requires timestamped evidence")
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("trigger evidence must be objects")
        _shared.require_text(item.get("fact"), "trigger evidence fact")
        if _shared.checked_time(item.get("time"), context, "trigger evidence time") > time:
            raise ValueError("trigger evidence cannot follow trigger_time")
    if task == "condition_agent":
        conditions = trigger.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("condition trigger requires all necessary conditions")
        for condition in conditions:
            if not isinstance(condition, dict) or condition.get("satisfied") is not True:
                raise ValueError("all necessary conditions must be satisfied")
            _shared.require_text(condition.get("condition"), "condition")
            _shared.require_text(condition.get("evidence"), "condition evidence")
            if _shared.checked_time(condition.get("time"), context, "condition time") > time:
                raise ValueError("condition evidence cannot follow trigger_time")
    elif task == "count_agent":
        if trigger.get("counting_mode") not in {"current", "cumulative", "distinct"}:
            raise ValueError("invalid counting_mode")
        threshold = _shared.require_count(trigger.get("threshold"), "threshold")
        previous = _shared.require_count(trigger.get("previous_count"), "previous_count")
        current = _shared.require_count(trigger.get("count"), "count")
        if not previous < threshold <= current:
            raise ValueError("count must cross the threshold from below")
        _shared.require_text(trigger.get("counting_basis"), "counting_basis")
        _shared.require_text(trigger.get("deduplication_rule"), "deduplication_rule")
    elif task == "target_agent":
        _shared.require_text(trigger.get("target"), "target")
        if trigger.get("attributes_match") is not True:
            raise ValueError("target attributes must match the user's specification")


def validate_result(result: dict[str, Any], context: GenerationContext, task: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("agent result must be an object")
    available = set(context.available_tools)
    catalog = set(TASK_TOOLS[task])
    if context.sample_type not in {"positive", "no_context", "missing_tool"} or not available <= catalog:
        raise ValueError("invalid agent generation context")
    if context.sample_type == "positive":
        if available != catalog or context.required_missing_tool is not None:
            raise ValueError("positive context must provide the task tools")
        if result.get("status") == "skip":
            return {"status": "skip", "reason": _shared.require_text(result.get("reason"), "reason")}
    else:
        if context.required_missing_tool not in catalog - available:
            raise ValueError("negative context must omit its required tool")
        if context.sample_type == "no_context" and available:
            raise ValueError("no_context must not expose tools")
        if context.sample_type == "missing_tool" and not available:
            raise ValueError("missing_tool must expose a partial tool catalog")
    if result.get("status") != "ok":
        raise ValueError("agent sample must have status ok; negative samples cannot skip")
    # Metadata is derived from the actual request, never invented by the generator.
    metadata = {
        "sample_type": context.sample_type,
        "available_tools": list(context.available_tools),
        "frontend_context": frontend_context(context),
    }
    for key, value in metadata.items():
        if key in result and result[key] != value:
            raise ValueError(f"model changed program-controlled {key}")
    work_type = result.get("work_type")
    if work_type not in {"tools", "reasoning", "tools_and_reasoning"}:
        raise ValueError("invalid work_type")
    required = _tool_list(result.get("required_tools"), "required_tools")
    if not required <= catalog:
        raise ValueError("required_tools must belong to the task tool catalog")
    if (work_type == "reasoning") != (not required):
        raise ValueError("pure reasoning requires no tools; tool work requires tools")
    if context.sample_type != "positive":
        if work_type == "reasoning" or context.required_missing_tool not in required:
            raise ValueError("negative task must require the missing tool; no reasoning negatives")
        _shared.require_text(result.get("question"), "question")
        question = _shared.checked_time(result.get("question_time"), context, "question_time")
        if (context.question_position == "start" and question != 0) or (
            context.question_position == "non_start" and question == 0
        ):
            raise ValueError("question_time conflicts with sampled position")
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError("negative sample requires exactly one refusal")
        refusal = responses[0]
        if not isinstance(refusal, dict) or set(refusal) != {"trigger_time", "response", "refusal_reason", "missing_tools"}:
            raise ValueError("refusal must not contain calls, results, or video-event claims")
        if _shared.checked_time(refusal["trigger_time"], context, "refusal time") != question:
            raise ValueError("refusal must occur immediately at question_time")
        _shared.require_text(refusal["response"], "refusal response")
        _shared.require_text(refusal["refusal_reason"], "refusal_reason")
        if _tool_list(refusal["missing_tools"], "missing_tools") != required - available:
            raise ValueError("refusal must identify exactly the missing required tools")
        if set(result) - {"status", "question", "question_time", "work_type", "required_tools", "responses", *metadata, "task", "video", "generation_context", "schema"}:
            raise ValueError("negative sample contains unsupported fields")
    else:
        if not required <= available:
            raise ValueError("positive task requires unavailable tools")
        _shared.validate_agent_result(result, context)
        response = result["responses"][0]
        _validate_trigger(response, context, task)
        call = response["messages"][0]["tool_calls"][0]
        arguments = json.loads(call["function"]["arguments"])
        suggested = _tool_list(arguments.get("suggested_capabilities"), "suggested_capabilities")
        if not suggested <= available or not required <= suggested:
            raise ValueError("delegation must cover required tools and cannot suggest unavailable tools")
        if work_type == "reasoning" and suggested:
            raise ValueError("pure reasoning must not suggest tools")
        envelope = _shared.agent_result_envelope(result)
        for key in ("observed_facts", "tool_information", "reasoning", "uncertainties"):
            values = envelope.get(key)
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise ValueError(f"backend {key} must be a string array")
        if work_type in {"reasoning", "tools_and_reasoning"} and not envelope["reasoning"]:
            raise ValueError("reasoning work must return an analysis")
        if work_type == "reasoning" and any(envelope[key] for key in ("sources", "actions", "tool_information")):
            raise ValueError("pure reasoning cannot claim external information or actions")
        for key in ("sources", "actions", "artifacts"):
            for item in envelope[key]:
                if not isinstance(item, dict):
                    raise ValueError(f"{key} entries must be objects")
                if key != "artifacts":
                    if item.get("tool") not in suggested:
                        raise ValueError("backend evidence must name an available delegated tool")
                    _shared.require_text(item.get("evidence"), f"{key} evidence")
                if key == "sources" and item["tool"] in WRITE_TOOLS:
                    raise ValueError("write tools require action evidence, not query sources")
                if key == "actions":
                    if item["tool"] not in WRITE_TOOLS:
                        raise ValueError("read-only tools cannot execute write actions")
                    if item.get("status") not in {"completed", "failed", "needs_confirmation"}:
                        raise ValueError("invalid action status")
                    _shared.require_text(item.get("operation"), "action operation")
        supported = {item["tool"] for item in envelope["sources"]}
        supported.update(item["tool"] for item in envelope["actions"] if item["status"] == "completed")
        if envelope["status"] == "success" and not required <= supported:
            raise ValueError("successful tool work requires evidence for every required tool")
        if envelope["status"] == "success" and any(item["status"] != "completed" for item in envelope["actions"]):
            raise ValueError("successful result cannot contain failed or unconfirmed actions")
    return {**result, **metadata}
