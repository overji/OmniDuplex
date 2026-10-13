"""Shared request construction and validation for realtime interaction tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import timestamp_seconds


_TRIGGER_PATTERN = re.compile(r"^(visual|sound|speech)(\+(visual|sound|speech))*$")


@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float
    preferred_modality: str
    question_position: str
    agent_delay_seconds: int | None = None
    initial_response_delay_seconds: int | None = 0


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty field: {name}")
    return value.strip()


def require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def checked_time(value: Any, context: GenerationContext, name: str) -> int:
    seconds = timestamp_seconds(value)
    if seconds > context.duration_seconds + 0.5:
        raise ValueError(f"{name} exceeds video duration: {value}")
    return seconds


def validate_common(
    result: dict[str, Any],
    context: GenerationContext,
    *,
    min_responses: int = 1,
    max_responses: int | None = None,
    max_words: int | None = None,
) -> tuple[dict[str, Any], int] | tuple[dict[str, str], None]:
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    if result.get("status") == "skip":
        reason = require_text(result.get("reason"), "reason")
        return {"status": "skip", "reason": reason}, None
    if result.get("status") != "ok":
        raise ValueError("status must be 'ok' or 'skip'")
    require_text(result.get("question"), "question")
    question_seconds = checked_time(
        result.get("question_time"), context, "question_time"
    )
    if context.question_position == "start" and question_seconds != 0:
        raise ValueError("question_time must be 00:00 for a start-position sample")
    if context.question_position == "non_start" and not 0 < question_seconds < context.duration_seconds:
        raise ValueError("question_time must be after 00:00 for a non-start sample")
    if context.initial_response_delay_seconds != 0:
        raise ValueError("v2 does not apply an initial silent interval")

    dependencies = []
    for key in ("audio_dependency", "vision_dependency"):
        value = result.get(key)
        if value not in {"required", "helpful", "none"}:
            raise ValueError(f"invalid {key}")
        dependencies.append(value)
    if "required" not in dependencies:
        raise ValueError("at least one dependency must be required")

    responses = result.get("responses")
    upper = max_responses if max_responses is not None else float("inf")
    if not isinstance(responses, list) or not min_responses <= len(responses) <= upper:
        label = f"{min_responses}+" if max_responses is None else f"{min_responses}-{max_responses}"
        raise ValueError(f"responses must contain {label} entries")
    previous = question_seconds - 1
    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"responses[{index}] must be an object")
        trigger_seconds = checked_time(
            response.get("trigger_time"), context, f"responses[{index}].trigger_time"
        )
        earliest = checked_time(response.get("window_start"), context, f"responses[{index}].window_start")
        latest = checked_time(response.get("window_end"), context, f"responses[{index}].window_end")
        if not question_seconds <= earliest <= trigger_seconds <= latest:
            raise ValueError("response time must lie within its valid trigger window")
        if trigger_seconds < question_seconds or trigger_seconds < previous:
            raise ValueError("responses must be chronological and not precede question_time")
        previous = trigger_seconds
        text = require_text(response.get("response"), f"responses[{index}].response")
        if max_words is not None and len(text.split()) > max_words:
            raise ValueError(f"responses[{index}].response exceeds {max_words} words")
        trigger_type = require_text(
            response.get("trigger_type"), f"responses[{index}].trigger_type"
        )
        if not _TRIGGER_PATTERN.fullmatch(trigger_type):
            raise ValueError(f"invalid trigger_type: {trigger_type!r}")
        modalities = set(trigger_type.split("+"))
        if result["audio_dependency"] == "required" and not (
            {"sound", "speech"} & modalities
        ):
            raise ValueError("trigger_type conflicts with required audio")
        if result["vision_dependency"] == "required" and "visual" not in modalities:
            raise ValueError("trigger_type conflicts with required vision")
        require_text(
            response.get("trigger_type_reason"),
            f"responses[{index}].trigger_type_reason",
        )
    return result, question_seconds


def validate_agent_result(
    result: dict[str, Any], context: GenerationContext, *, allow_legacy_agent_windows=False
) -> dict[str, Any]:
    if result.get("status") == "skip":
        reason = require_text(result.get("reason"), "reason")
        return {"status": "skip", "reason": reason}
    if context.agent_delay_seconds is None:
        raise ValueError("video has no legal duration-based backend delay; return skip")
    validate_common(result, context, min_responses=1)
    for response in result["responses"]:
        validate_agent_response(response, context, allow_legacy_agent_windows=allow_legacy_agent_windows)
    return result


def validate_agent_response(response: dict[str, Any], context: GenerationContext,
                            *, allow_legacy_agent_windows=False) -> None:
    trigger_seconds = timestamp_seconds(response["trigger_time"])
    messages = response.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("agent response must contain exactly three messages")
    assistant_call, tool_result, assistant_final = messages
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("agent messages must be objects")
    legacy_final = (allow_legacy_agent_windows and "window_start" not in assistant_final
                    and "window_end" not in assistant_final)
    for message, allowed in zip(messages, (
        {"role", "timestamp", "content", "tool_calls"},
        {"role", "name", "timestamp", "content"},
        {"role", "timestamp", "content", "outcome"} | (set() if legacy_final else {"window_start", "window_end"}),
    )):
        if set(message) != allowed:
            raise ValueError("agent message fields must match its role")
    if assistant_call.get("role") != "assistant":
        raise ValueError("first agent message must be assistant")
    if checked_time(assistant_call.get("timestamp"), context, "call timestamp") != trigger_seconds:
        raise ValueError("acknowledgement and agent_call must occur at trigger_time")
    acknowledgement = require_text(
        assistant_call.get("content"), "frontend acknowledgement"
    )
    if len(acknowledgement.split()) > 15:
        raise ValueError("frontend acknowledgement must be at most 15 words")
    if response["response"] != acknowledgement:
        raise ValueError("response must equal the immediate frontend acknowledgement")
    calls = assistant_call.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("assistant must make exactly one tool call")
    call = calls[0]
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ValueError("tool call must use function type")
    if "id" in call or "tool_call_id" in call:
        raise ValueError("tool calls must not contain association IDs")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "agent_call":
        raise ValueError("tool function must be agent_call")
    arguments_text = require_text(function.get("arguments"), "agent_call arguments")
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise ValueError("agent_call arguments must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("agent_call arguments must decode to an object")
    required_arguments = {
        "instruction", "observed_context", "user_goal", "constraints_and_authorization",
        "suggested_capabilities", "expected_output", "extracted_parameters",
    }
    if set(arguments) != required_arguments:
        raise ValueError("agent_call arguments must match the function schema")
    for key in (
        "instruction",
        "observed_context",
        "user_goal",
        "constraints_and_authorization",
        "suggested_capabilities",
        "expected_output",
    ):
        value = arguments.get(key)
        if key == "suggested_capabilities":
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError("suggested_capabilities must be a string list")
        else:
            require_text(value, f"agent_call.{key}")

    parameters = arguments.get("extracted_parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("extracted_parameters must be a non-empty array")
    for parameter in parameters:
        if not isinstance(parameter, dict) or set(parameter) != {"name", "value", "evidence_time", "evidence"}:
            raise ValueError("extracted_parameters entries must be objects")
        for key in ("name", "value", "evidence"):
            require_text(parameter.get(key), f"extracted_parameters.{key}")
        evidence_time = checked_time(parameter.get("evidence_time"), context, "evidence_time")
        if evidence_time > timestamp_seconds(response["window_start"]):
            raise ValueError("parameter evidence must be available at the earliest eligible call time")

    if timestamp_seconds(response["window_end"]) + context.agent_delay_seconds > context.duration_seconds + 0.5:
        raise ValueError("latest eligible call cannot finish within video duration")
    result_seconds = trigger_seconds + context.agent_delay_seconds
    if result_seconds > context.duration_seconds + 0.5:
        raise ValueError("agent result exceeds video duration")
    if tool_result.get("role") != "tool" or tool_result.get("name") != "agent_call":
        raise ValueError("second message must be the agent_call tool result")
    if any("tool_call_id" in message or "id" in message for message in messages):
        raise ValueError("agent messages must not contain association IDs")
    if checked_time(tool_result.get("timestamp"), context, "result timestamp") != result_seconds:
        raise ValueError("tool result timestamp does not match sampled delay")
    result_content = require_text(tool_result.get("content"), "simulated result")
    try:
        envelope = json.loads(result_content)
    except json.JSONDecodeError as exc:
        raise ValueError("tool result content must be valid JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("simulated") is not True:
        raise ValueError("tool result must be explicitly simulated")
    if envelope.get("status") not in {"success", "needs_confirmation", "failed"}:
        raise ValueError("invalid simulated result status")
    require_text(envelope.get("summary"), "simulated result summary")
    for key in ("sources", "actions", "artifacts"):
        if not isinstance(envelope.get(key), list):
            raise ValueError(f"simulated result {key} must be an array")

    if assistant_final.get("role") != "assistant":
        raise ValueError("third message must be the frontend assistant result")
    final_time = checked_time(assistant_final.get("timestamp"), context, "final timestamp")
    if legacy_final:
        if final_time != result_seconds:
            raise ValueError("legacy frontend final must coincide with the tool result")
    else:
        start = checked_time(assistant_final["window_start"], context, "final window_start")
        end = checked_time(assistant_final["window_end"], context, "final window_end")
        if not result_seconds + 1 <= start <= final_time <= end <= int(context.duration_seconds):
            raise ValueError("frontend final must lie in its own window strictly after the tool result")
        if timestamp_seconds(response["window_end"]) + context.agent_delay_seconds + 1 > int(context.duration_seconds):
            raise ValueError("latest eligible call must leave time for a post-result reply")
    frontend_final = require_text(assistant_final.get("content"), "frontend final")
    if len(frontend_final.split()) > 40:
        raise ValueError("frontend final must be at most 40 words")
    if assistant_final.get("tool_calls"):
        raise ValueError("frontend final must not make another tool call")


def agent_result_envelope(response: dict[str, Any]) -> dict[str, Any]:
    """Return a validated agent trace's decoded simulated-result envelope."""
    content = response["messages"][1]["content"]
    envelope = json.loads(content)
    assert isinstance(envelope, dict)
    return envelope
