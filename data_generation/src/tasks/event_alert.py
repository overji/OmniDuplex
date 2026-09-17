"""Instant Event Alert task prompt construction and result validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import (
    format_mmss,
    load_json_object,
    timestamp_seconds,
)
from ..config import (
    choose_initial_response_delay,
    choose_preferred_modality,
    choose_question_position,
)
from ..video_caption import probe_video_duration


PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "task_generation"
    / "proactive_alert"
    / "event_alert.md"
)
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float
    preferred_modality: str
    question_position: str
    initial_response_delay_seconds: int | None = 0


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def _load_evidence(path: Path) -> dict[str, Any]:
    source = load_json_object(path)
    audio_events = source.get("audio_events")
    visual_events = source.get("visual_events")
    if not isinstance(audio_events, list) or not isinstance(visual_events, list):
        raise ValueError("input must contain audio_events and visual_events arrays")

    evidence = {
        "audio_events": audio_events,
        "visual_events": visual_events,
    }
    if "uncertainties" in source:
        evidence["event_uncertainties"] = source["uncertainties"]
    return evidence


def _render_system_prompt(context: GenerationContext) -> str:
    preferred = context.preferred_modality
    fallback = "vision" if preferred == "audio" else "audio"
    if context.question_position == "start":
        question_policy = (
            "For this sample, question_time MUST be 00:00. Include every target "
            "occurrence at or after 00:00 as an event response."
        )
    elif context.question_position == "non_start":
        question_policy = (
            "For this sample, choose a natural question_time strictly after 00:00 "
            f"and no later than {format_mmss(context.duration_seconds)}. It MUST NOT "
            "equal any onset of the chosen target event. When the target repeats, "
            "prefer a time between occurrences so the sample contains both an "
            "immediate historical summary and later event alerts."
        )
    else:
        raise ValueError(f"invalid question position: {context.question_position!r}")
    if context.initial_response_delay_seconds is None:
        question_policy += " The video is too short for the required silent interval; return skip."
    else:
        question_policy += (
            f" Keep silent after question_time for at least {context.initial_response_delay_seconds} "
            "seconds (you may wait longer); the first response must be no earlier than "
            "question_time plus this interval. If no response fits, return skip."
        )

    replacements = {
        "{{VIDEO_DURATION}}": format_mmss(context.duration_seconds),
        "{{PREFERRED_MODALITY}}": preferred,
        "{{PREFERRED_DEPENDENCY}}": f"{preferred}_dependency",
        "{{FALLBACK_MODALITY}}": fallback,
        "{{FALLBACK_DEPENDENCY}}": f"{fallback}_dependency",
        "{{QUESTION_TIME_POLICY}}": question_policy,
    }
    prompt = PROMPT_TEMPLATE
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
    *,
    objects_path: Path | None = None,
) -> GenerationRequest:
    del objects_path
    evidence = _load_evidence(path)
    resolved_video_path = video_path.expanduser().resolve()
    duration_seconds = probe_video_duration(resolved_video_path)
    preferred_modality = choose_preferred_modality()
    question_position = choose_question_position(duration_seconds)
    context = GenerationContext(
        duration_seconds=duration_seconds,
        preferred_modality=preferred_modality,
        question_position=question_position,
        initial_response_delay_seconds=choose_initial_response_delay(duration_seconds),
    )
    return GenerationRequest(
        system_prompt=_render_system_prompt(context),
        user_prompt=json.dumps(evidence, ensure_ascii=False, indent=2),
        context=context,
        video_path=resolved_video_path,
    )


def _validate_dependencies(result: dict[str, Any]) -> None:
    allowed = {"required", "helpful", "none"}
    for key in ("audio_dependency", "vision_dependency"):
        if result[key] not in allowed:
            raise ValueError(f"invalid {key}")
    required_count = sum(
        result[key] == "required"
        for key in ("audio_dependency", "vision_dependency")
    )
    if required_count != 1:
        raise ValueError(
            "exactly one of audio_dependency and vision_dependency must be 'required'"
        )


def validate_result(
    result: dict[str, Any], context: GenerationContext
) -> dict[str, Any]:
    status = result.get("status")
    if status == "skip":
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip result must contain a non-empty reason")
        return {"status": "skip", "reason": reason.strip()}
    if status != "ok":
        raise ValueError("status must be 'ok' or 'skip'")

    for key in (
        "question",
        "question_time",
        "audio_dependency",
        "vision_dependency",
    ):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"missing or empty field: {key}")

    question_seconds = timestamp_seconds(result["question_time"])
    if question_seconds > context.duration_seconds + 0.5:
        raise ValueError("question_time exceeds video duration")
    if context.question_position == "start" and question_seconds != 0:
        raise ValueError("question_time must be 00:00 for a start-position sample")
    if context.question_position == "non_start" and question_seconds == 0:
        raise ValueError("question_time must be after 00:00 for a non-start sample")
    if context.initial_response_delay_seconds is None:
        raise ValueError("video is too short for the required initial response delay")

    _validate_dependencies(result)
    responses = result.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("ok result must contain at least one response")

    response_fields = (
        "trigger_time",
        "response",
        "trigger_type",
        "trigger_type_reason",
        "event_description",
    )
    trigger_pattern = re.compile(r"^(visual|sound|speech)(\+(visual|sound|speech))*$")
    seen_trigger_times: set[str] = set()
    seen_event_times: set[str] = set()
    historical_response_count = 0

    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"responses[{index}] must be an object")
        for key in response_fields:
            if not isinstance(response.get(key), str) or not response[key].strip():
                raise ValueError(f"missing or empty responses[{index}].{key}")

        trigger_time = response["trigger_time"]
        trigger_seconds = timestamp_seconds(trigger_time)
        if trigger_seconds > context.duration_seconds + 0.5:
            raise ValueError(f"trigger_time exceeds video duration: {trigger_time}")
        if trigger_seconds < question_seconds:
            raise ValueError(f"response precedes question_time: {trigger_time}")
        if index == 0 and trigger_seconds < question_seconds + context.initial_response_delay_seconds:
            raise ValueError("first response occurs before the sampled silent interval")
        if trigger_time in seen_trigger_times:
            raise ValueError(f"duplicate trigger_time: {trigger_time}")
        seen_trigger_times.add(trigger_time)

        trigger_type = response["trigger_type"]
        if not trigger_pattern.fullmatch(trigger_type):
            raise ValueError(f"invalid trigger_type: {trigger_type!r}")
        trigger_modalities = set(trigger_type.split("+"))
        if result["audio_dependency"] == "required" and not (
            {"sound", "speech"} & trigger_modalities
        ):
            raise ValueError(
                f"responses[{index}].trigger_type conflicts with required audio"
            )
        if (
            result["vision_dependency"] == "required"
            and "visual" not in trigger_modalities
        ):
            raise ValueError(
                f"responses[{index}].trigger_type conflicts with required vision"
            )

        event_times = response.get("event_times")
        if not isinstance(event_times, list) or not event_times:
            raise ValueError(f"responses[{index}].event_times must be a non-empty list")
        event_seconds: list[int] = []
        for event_time in event_times:
            seconds = timestamp_seconds(event_time)
            if seconds > context.duration_seconds + 0.5:
                raise ValueError(f"event_time exceeds video duration: {event_time}")
            if event_time in seen_event_times:
                raise ValueError(f"duplicate event_time: {event_time}")
            seen_event_times.add(event_time)
            event_seconds.append(seconds)
        if event_seconds != sorted(event_seconds) or len(set(event_seconds)) != len(
            event_seconds
        ):
            raise ValueError(f"responses[{index}].event_times must be sorted and unique")

        has_past = any(seconds < question_seconds for seconds in event_seconds)
        has_current_or_future = any(
            seconds >= question_seconds for seconds in event_seconds
        )
        if has_past and has_current_or_future:
            raise ValueError("a response cannot mix historical and future event_times")
        if has_past:
            historical_response_count += 1
            if trigger_seconds < question_seconds + context.initial_response_delay_seconds:
                raise ValueError("historical events must trigger a delayed response")
        else:
            if len(event_seconds) != 1:
                raise ValueError("each current or future event needs its own response")
            if (
                context.question_position == "non_start"
                and event_seconds[0] == question_seconds
            ):
                raise ValueError("non-start question_time cannot equal an event onset")
            if trigger_seconds != event_seconds[0]:
                raise ValueError("future response trigger_time must equal its event_time")

    if historical_response_count > 1:
        raise ValueError("all historical occurrences must use one summary response")

    responses.sort(key=lambda item: timestamp_seconds(item["trigger_time"]))
    return result
