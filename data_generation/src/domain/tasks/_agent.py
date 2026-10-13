"""Program-controlled capability contexts and triggered agent sample contracts."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ..agent_tools import TOOLS, WRITE_TOOLS, describe_tool, validate_arguments
from ..common import timestamp_seconds
from ..config import AGENT_REFUSAL_WINDOW_SECONDS
from . import _shared


@dataclass(frozen=True)
class GenerationContext(_shared.GenerationContext):
    sample_type: str = "positive"
    available_tools: tuple[str, ...] = ()
    required_missing_tool: str | None = None


GenerationRequest = _shared.GenerationRequest


def frontend_context(context: GenerationContext, capabilities_template: str) -> dict[str, Any]:
    if context.sample_type == "no_context":
        return {"system_prompt": "You are an audiovisual assistant. State capability limitations honestly.", "tools": []}
    text = capabilities_template
    prefix, definition = text.split("Frontend function definition:\n", 1)
    available = "\n".join(f"- {name}: {describe_tool(name)}" for name in context.available_tools)
    return {
        "system_prompt": prefix.replace("{{AVAILABLE_TOOLS}}", available).strip(),
        "tools": [json.loads(definition)],
    }


def _tool_list(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{label} must be a string array")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique tool names")
    return set(value)


def validate_result(result: dict[str, Any], context: GenerationContext, capabilities_template: str,
                    *, allow_legacy_agent_windows=False) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("agent result must be an object")
    available = set(context.available_tools)
    catalog = set(TOOLS)
    if context.sample_type not in {"positive", "no_context", "missing_tool"} or not available <= catalog:
        raise ValueError("invalid agent generation context")
    if context.sample_type == "positive":
        if available != catalog or context.required_missing_tool is not None:
            raise ValueError("positive context must provide the shared tools")
    else:
        if context.required_missing_tool not in catalog - available:
            raise ValueError("negative context must omit its required tool")
        if context.sample_type == "no_context" and available:
            raise ValueError("no_context must not expose tools")
        if context.sample_type == "missing_tool" and available != catalog - {context.required_missing_tool}:
            raise ValueError("missing_tool must omit exactly its sampled tool")
    if result.get("status") == "skip":
        return {"status": "skip", "reason": _shared.require_text(result.get("reason"), "reason")}
    if result.get("status") != "ok":
        raise ValueError("agent sample must have status ok; unsupported status")
    # Metadata is derived from the actual request, never invented by the generator.
    metadata = {
        "sample_type": context.sample_type,
        "available_tools": list(context.available_tools),
        "frontend_context": frontend_context(context, capabilities_template),
    }
    for key, value in metadata.items():
        if key in result and result[key] != value:
            raise ValueError(f"model changed program-controlled {key}")
    work_type = result.get("work_type")
    if work_type not in {"tools", "tools_and_reasoning"}:
        raise ValueError("Agent tasks require tools or tools_and_reasoning; pure reasoning is not eligible")
    required = _tool_list(result.get("required_tools"), "required_tools")
    if not required <= catalog:
        raise ValueError("required_tools must belong to the shared tool catalog")
    if not required:
        raise ValueError("Agent tasks require at least one necessary backend tool")
    if context.sample_type != "positive":
        if context.required_missing_tool not in required:
            raise ValueError("negative task must require the missing tool; no reasoning negatives")
        _shared.require_text(result.get("question"), "question")
        question = _shared.checked_time(result.get("question_time"), context, "question_time")
        if (context.question_position == "start" and question != 0) or (
            context.question_position == "non_start" and not 0 < question < context.duration_seconds
        ):
            raise ValueError("question_time conflicts with sampled position")
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != 1:
            raise ValueError("negative sample requires exactly one refusal")
        refusal = responses[0]
        if not isinstance(refusal, dict) or set(refusal) != {"window_start", "trigger_time", "window_end", "response", "refusal_reason", "missing_tools"}:
            raise ValueError("refusal must not contain calls, results, or video-event claims")
        if _shared.checked_time(refusal["trigger_time"], context, "refusal time") != question:
            raise ValueError("refusal must occur immediately at question_time")
        start = _shared.checked_time(refusal["window_start"], context, "window_start")
        end = _shared.checked_time(refusal["window_end"], context, "window_end")
        legacy = allow_legacy_agent_windows and start == end == question
        if not legacy and (start != question or end != question + AGENT_REFUSAL_WINDOW_SECONDS
                           or end > math.floor(context.duration_seconds)):
            raise ValueError("refusal window must span question_time through question_time + 3 within video")
        _shared.require_text(refusal["response"], "refusal response")
        _shared.require_text(refusal["refusal_reason"], "refusal_reason")
        if _tool_list(refusal["missing_tools"], "missing_tools") != required - available:
            raise ValueError("refusal must identify exactly the missing required tools")
        if set(result) - {"status", "question", "question_time", "work_type", "required_tools", "responses", *metadata, "task", "video", "generation_context", "schema", "ability", "mode", "annotations", "audio_dependency", "vision_dependency"}:
            raise ValueError("negative sample contains unsupported fields")
    else:
        if not required <= available:
            raise ValueError("positive task requires unavailable tools")
        _shared.validate_agent_result(result, context, allow_legacy_agent_windows=allow_legacy_agent_windows)
        for response in result["responses"]:
            call = response["messages"][0]["tool_calls"][0]
            arguments = json.loads(call["function"]["arguments"])
            suggested = _tool_list(arguments.get("suggested_capabilities"), "suggested_capabilities")
            if not suggested <= available or not required <= suggested:
                raise ValueError("delegation must cover required tools and cannot suggest unavailable tools")
            envelope = _shared.agent_result_envelope(response)
            for key in ("observed_facts", "tool_information", "reasoning", "uncertainties"):
                values = envelope.get(key)
                if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                    raise ValueError(f"backend {key} must be a string array")
            if work_type == "tools_and_reasoning" and not envelope["reasoning"]:
                raise ValueError("reasoning work must return an analysis")
            for key in ("sources", "actions", "artifacts"):
                for item in envelope[key]:
                    if not isinstance(item, dict):
                        raise ValueError(f"{key} entries must be objects")
                    if key != "artifacts":
                        if item.get("tool") not in suggested:
                            raise ValueError("backend evidence must name an available delegated tool")
                        _shared.require_text(item.get("evidence"), f"{key} evidence")
                        validate_arguments(item["tool"], item.get("arguments"))
                        if item["tool"] == "video_segment_query":
                            start = _shared.checked_time(item["arguments"]["start"], context, "segment start")
                            end = _shared.checked_time(item["arguments"]["end"], context, "segment end")
                            if not start <= end <= timestamp_seconds(response["window_start"]):
                                raise ValueError("video query cannot access future footage")
                    if key == "sources" and item["tool"] in WRITE_TOOLS:
                        raise ValueError("write tools require action evidence, not query sources")
                    if key == "actions":
                        if item["tool"] not in WRITE_TOOLS:
                            raise ValueError("read-only tools cannot execute write actions")
                        if item.get("status") not in {"completed", "failed", "needs_confirmation"}:
                            raise ValueError("invalid action status")
                        _shared.require_text(item.get("operation"), "action operation")
            if response["messages"][2].get("outcome") != envelope["status"]:
                raise ValueError("final outcome must match backend status")
            supported = {item["tool"] for item in envelope["sources"]}
            supported.update(item["tool"] for item in envelope["actions"] if item["status"] == "completed")
            if envelope["status"] == "success" and not required <= supported:
                raise ValueError("successful tool work requires evidence for every required tool")
            if envelope["status"] == "success" and any(item["status"] != "completed" for item in envelope["actions"]):
                raise ValueError("successful result cannot contain failed or unconfirmed actions")
    return {**result, **metadata}
