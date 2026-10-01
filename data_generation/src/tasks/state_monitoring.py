"""Realtime State Monitor task prompt construction and result validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import format_mmss, timestamp_seconds
from ..config import choose_initial_response_delay, choose_question_position
from ..video_caption import load_caption, probe_video_duration


PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "task_generation"
    / "narration"
    / "state_monitoring.md"
)
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float
    question_position: str
    initial_response_delay_seconds: int | None = 0


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def _render_system_prompt(context: GenerationContext) -> str:
    if context.question_position == "start":
        question_policy = "For this sample, question_time MUST be 00:00."
    elif context.question_position == "non_start":
        question_policy = (
            "For this sample, choose a natural question_time strictly after 00:00 "
            f"and before {format_mmss(context.duration_seconds)}. Choose it early "
            "enough to capture the selected state's later transitions."
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
        "{{VIDEO_DURATION_SECONDS}}": f"{context.duration_seconds:.0f}",
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
) -> GenerationRequest:
    resolved_video_path = video_path.expanduser().resolve()
    duration_seconds = probe_video_duration(resolved_video_path)
    evidence = {"caption": load_caption(path, duration_seconds)}
    context = GenerationContext(
        duration_seconds=duration_seconds,
        question_position=choose_question_position(duration_seconds),
        initial_response_delay_seconds=choose_initial_response_delay(duration_seconds),
    )
    return GenerationRequest(
        system_prompt=_render_system_prompt(context),
        user_prompt=json.dumps(evidence, ensure_ascii=False, indent=2),
        context=context,
        video_path=resolved_video_path,
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

    for key in ("audio_scan", "question", "question_time", "audio_dependency"):
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

    audio_dependency = result["audio_dependency"]
    if audio_dependency not in {"required", "helpful", "none"}:
        raise ValueError("invalid audio_dependency")

    responses = result.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("responses must contain at least one transition")

    response_fields = (
        "trigger_time",
        "response",
        "trigger_type",
        "trigger_type_reason",
        "event_description",
    )
    trigger_pattern = re.compile(r"^(visual|sound|speech)(\+(visual|sound|speech))*$")
    transition_pattern = re.compile(r"\bfrom\b.+\bto\b", re.IGNORECASE)
    previous_trigger_seconds = question_seconds
    seen_trigger_times: set[str] = set()

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
        if trigger_seconds <= question_seconds:
            raise ValueError("state transition must occur after question_time")
        if index == 0 and trigger_seconds < question_seconds + context.initial_response_delay_seconds:
            raise ValueError("first response occurs before the sampled silent interval")
        if trigger_seconds <= previous_trigger_seconds:
            raise ValueError("responses must be strictly chronological")
        if trigger_time in seen_trigger_times:
            raise ValueError(f"duplicate trigger_time: {trigger_time}")
        previous_trigger_seconds = trigger_seconds
        seen_trigger_times.add(trigger_time)

        response_text = response["response"]
        if len(response_text.split()) > 15:
            raise ValueError(f"responses[{index}].response exceeds 15 words")
        if not transition_pattern.search(response_text):
            raise ValueError(
                f"responses[{index}].response must name a transition from X to Y"
            )

        trigger_type = response["trigger_type"]
        if not trigger_pattern.fullmatch(trigger_type):
            raise ValueError(f"invalid trigger_type: {trigger_type!r}")
        trigger_modalities = set(trigger_type.split("+"))
        if audio_dependency == "required" and not (
            {"sound", "speech"} & trigger_modalities
        ):
            raise ValueError(
                f"responses[{index}].trigger_type conflicts with required audio"
            )
        if audio_dependency in {"helpful", "none"} and "visual" not in trigger_modalities:
            raise ValueError(
                f"responses[{index}].trigger_type must include visual for a visual target"
            )

    return result
