"""Shared request construction and validation for realtime interaction tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import format_mmss, load_json_object, timestamp_seconds
from ..config import (
    choose_agent_delay,
    choose_preferred_modality,
    choose_question_position,
)
from ..video_caption import probe_video_duration


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "task_generation"
AGENT_CAPABILITIES_PATH = PROMPT_ROOT / "agent" / "capabilities.md"
_TRIGGER_PATTERN = re.compile(r"^(visual|sound|speech)(\+(visual|sound|speech))*$")


@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float
    preferred_modality: str
    question_position: str
    agent_delay_seconds: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def load_evidence(path: Path, objects_path: Path | None = None) -> dict[str, Any]:
    source = load_json_object(path)
    evidence: dict[str, Any] = {}
    for key in ("audio_events", "visual_events"):
        value = source.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, dict) for item in value
        ):
            raise ValueError(f"input must contain an array of objects: {key}")
        evidence[key] = value
    if objects_path is not None:
        object_source = load_json_object(objects_path)
        objects = object_source.get("items")
        if not isinstance(objects, list) or not all(
            isinstance(item, dict) for item in objects
        ):
            raise ValueError("objects input must contain an items array of objects")
        evidence["items"] = objects
        if "uncertainties" in object_source:
            evidence["object_uncertainties"] = object_source["uncertainties"]
    if "uncertainties" in source:
        evidence["event_uncertainties"] = source["uncertainties"]
    return evidence


def _question_policy(context: GenerationContext, *, historical: bool) -> str:
    if context.question_position == "start":
        return "For this sample, question_time MUST be 00:00."
    if context.question_position != "non_start":
        raise ValueError(f"invalid question position: {context.question_position!r}")
    policy = (
        "For this sample, choose a natural question_time strictly after 00:00 and "
        f"before {format_mmss(context.duration_seconds)}."
    )
    if historical:
        return policy + (
            " Prefer a time between occurrences so historical and later evidence "
            "can both be represented; purely historical or future samples are allowed. "
            "question_time MUST NOT equal a counted event time or first appearance."
        )
    return policy + " Only create task responses at or after question_time."


def render_prompt(
    prompt_path: Path,
    context: GenerationContext,
    *,
    historical: bool = False,
    agent: bool = False,
) -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    fallback = "vision" if context.preferred_modality == "audio" else "audio"
    replacements = {
        "{{VIDEO_DURATION}}": format_mmss(context.duration_seconds),
        "{{VIDEO_DURATION_SECONDS}}": f"{context.duration_seconds:.0f}",
        "{{PREFERRED_MODALITY}}": context.preferred_modality,
        "{{FALLBACK_MODALITY}}": fallback,
        "{{QUESTION_TIME_POLICY}}": _question_policy(
            context, historical=historical
        ),
    }
    if agent:
        capabilities = AGENT_CAPABILITIES_PATH.read_text(encoding="utf-8").strip()
        if context.agent_delay_seconds is None:
            timing = (
                "The video is shorter than 15 seconds, so no legal backend delay "
                "exists. Return skip."
            )
        else:
            timing = (
                f"The simulated backend delay is exactly "
                f"{context.agent_delay_seconds} seconds. Select a trigger early enough "
                "that trigger_time plus this delay does not exceed the video duration."
            )
        replacements.update(
            {
                "{{AGENT_CAPABILITIES}}": capabilities,
                "{{AGENT_TIMING_POLICY}}": timing,
            }
        )
    for placeholder, value in replacements.items():
        if placeholder not in prompt:
            raise ValueError(f"missing prompt placeholder: {placeholder}")
        prompt = prompt.replace(placeholder, value)
    unresolved = re.findall(r"\{\{[A-Z_]+\}\}", prompt)
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {unresolved}")
    return prompt


def build_request(
    path: Path,
    video_path: Path,
    prompt_path: Path,
    *,
    objects_path: Path | None = None,
    historical: bool = False,
    agent: bool = False,
) -> GenerationRequest:
    evidence = load_evidence(path, objects_path)
    resolved_video_path = video_path.expanduser().resolve()
    duration_seconds = probe_video_duration(resolved_video_path)
    context = GenerationContext(
        duration_seconds=duration_seconds,
        preferred_modality=choose_preferred_modality(),
        question_position=choose_question_position(duration_seconds),
        agent_delay_seconds=(
            choose_agent_delay(duration_seconds) if agent else None
        ),
    )
    return GenerationRequest(
        system_prompt=render_prompt(
            prompt_path, context, historical=historical, agent=agent
        ),
        user_prompt=json.dumps(evidence, ensure_ascii=False, indent=2),
        context=context,
        video_path=resolved_video_path,
    )


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
    if context.question_position == "non_start" and question_seconds == 0:
        raise ValueError("question_time must be after 00:00 for a non-start sample")

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
        require_text(
            response.get("event_description"),
            f"responses[{index}].event_description",
        )
    return result, question_seconds


def validate_counting_history(
    result: dict[str, Any], context: GenerationContext, *, deduplicated: bool
) -> dict[str, Any]:
    validated, question_seconds = validate_common(
        result, context, min_responses=1, max_words=20
    )
    if question_seconds is None:
        return validated
    seen_times: set[str] = set()
    seen_ids: set[str] = set()
    running_count = 0
    historical_groups = 0
    for index, response in enumerate(result["responses"]):
        trigger_seconds = timestamp_seconds(response["trigger_time"])
        event_times = response.get("event_times")
        if not isinstance(event_times, list) or not event_times:
            raise ValueError(f"responses[{index}].event_times must be non-empty")
        parsed = [checked_time(value, context, "event_time") for value in event_times]
        if parsed != sorted(parsed) or any(value in seen_times for value in event_times):
            raise ValueError("event_times must be globally chronological and unique")
        seen_times.update(event_times)
        past = [value < question_seconds for value in parsed]
        if any(past):
            historical_groups += 1
            if not all(past) or trigger_seconds != question_seconds:
                raise ValueError("historical events require one immediate summary")
        elif len(parsed) != 1 or trigger_seconds != parsed[0]:
            raise ValueError("each later event needs its own response at its event_time")
        elif context.question_position == "non_start" and parsed[0] == question_seconds:
            raise ValueError("non-start question_time cannot equal a counted event time")

        increment = len(parsed)
        if deduplicated:
            targets = response.get("new_targets")
            if not isinstance(targets, list) or len(targets) != len(parsed):
                raise ValueError("new_targets must correspond one-to-one with event_times")
            for target in targets:
                if not isinstance(target, dict):
                    raise ValueError("new_targets entries must be objects")
                target_id = require_text(target.get("target_id"), "target_id")
                require_text(target.get("name"), "target name")
                require_text(target.get("evidence"), "target evidence")
                if target_id in seen_ids:
                    raise ValueError("duplicate target ID must not increase count")
                seen_ids.add(target_id)
        running_count += increment
        if require_count(response.get("count"), f"responses[{index}].count") != running_count:
            raise ValueError("count must equal cumulative event_times")
    if historical_groups > 1:
        raise ValueError("all historical events must share one response")
    return result


def validate_agent_result(
    result: dict[str, Any], context: GenerationContext
) -> dict[str, Any]:
    if result.get("status") == "skip":
        reason = require_text(result.get("reason"), "reason")
        return {"status": "skip", "reason": reason}
    if context.agent_delay_seconds is None:
        raise ValueError("videos shorter than 15 seconds must return skip")
    # Agent responses use messages instead of the direct-response field.
    shadow = dict(result)
    responses = result.get("responses")
    if isinstance(responses, list):
        shadow["responses"] = [
            {**item, "response": "agent interaction"}
            if isinstance(item, dict)
            else item
            for item in responses
        ]
    _, question_seconds = validate_common(
        shadow, context, min_responses=1, max_responses=1
    )
    assert question_seconds is not None
    response = result["responses"][0]
    trigger_seconds = timestamp_seconds(response["trigger_time"])
    messages = response.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("agent response must contain exactly three messages")
    assistant_call, tool_result, assistant_final = messages
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("agent messages must be objects")
    if assistant_call.get("role") != "assistant":
        raise ValueError("first agent message must be assistant")
    if checked_time(assistant_call.get("timestamp"), context, "call timestamp") != trigger_seconds:
        raise ValueError("acknowledgement and agent_call must occur at trigger_time")
    acknowledgement = require_text(
        assistant_call.get("content"), "frontend acknowledgement"
    )
    if len(acknowledgement.split()) > 15:
        raise ValueError("frontend acknowledgement must be at most 15 words")
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
            if not isinstance(value, list) or not value or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError("suggested_capabilities must be a non-empty string list")
        else:
            require_text(value, f"agent_call.{key}")

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
    if checked_time(assistant_final.get("timestamp"), context, "final timestamp") != result_seconds:
        raise ValueError("frontend final must coincide with the tool result")
    frontend_final = require_text(assistant_final.get("content"), "frontend final")
    if len(frontend_final.split()) > 40:
        raise ValueError("frontend final must be at most 40 words")
    if assistant_final.get("tool_calls"):
        raise ValueError("frontend final must not make another tool call")
    return result


def agent_result_envelope(result: dict[str, Any]) -> dict[str, Any]:
    """Return a validated agent trace's decoded simulated-result envelope."""
    content = result["responses"][0]["messages"][1]["content"]
    envelope = json.loads(content)
    assert isinstance(envelope, dict)
    return envelope
