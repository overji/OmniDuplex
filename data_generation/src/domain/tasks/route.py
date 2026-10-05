"""Video-to-task routing prompt construction and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..catalog import ABILITIES

@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def build_request(video_path: Path, duration: float, caption: list, prompt: str) -> GenerationRequest:
    evidence = {"caption": caption}
    catalog = "\n".join(f"- {name}: {spec.description}" for name, spec in ABILITIES.items())
    prompt = prompt.strip()
    prompt = prompt.replace("{{VIDEO_DURATION}}", f"{duration:.0f} seconds")
    prompt = prompt.replace("{{TASK_CATALOG}}", catalog)
    if "{{" in prompt:
        raise ValueError("unresolved route prompt placeholder")
    return GenerationRequest(
        system_prompt=prompt,
        user_prompt=json.dumps(evidence, ensure_ascii=False, indent=2),
        context=GenerationContext(duration),
        video_path=video_path,
    )


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("route result must be an object")
    if result.get("status") == "skip":
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip result must contain a non-empty reason")
        return {"status": "skip", "reason": reason.strip()}
    if result.get("status") != "ok":
        raise ValueError("status must be 'ok' or 'skip'")
    unsuitable = result.get("unsuitable_tasks")
    ranked = result.get("ranked_tasks")
    if not isinstance(unsuitable, list) or not isinstance(ranked, list) or not ranked:
        raise ValueError("route result requires unsuitable_tasks and ranked_tasks")
    seen: set[str] = set()
    for item in unsuitable:
        if not isinstance(item, dict) or item.get("task") not in ABILITIES:
            raise ValueError("invalid unsuitable task")
        task = item["task"]
        if task in seen:
            raise ValueError("duplicate routed task")
        seen.add(task)
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("unsuitable task reason must be non-empty")
    for expected_rank, item in enumerate(ranked, 1):
        if not isinstance(item, dict) or item.get("task") not in ABILITIES:
            raise ValueError("invalid ranked task")
        task = item["task"]
        if task in seen:
            raise ValueError("duplicate routed task")
        seen.add(task)
        if item.get("rank") != expected_rank:
            raise ValueError("ranks must be consecutive")
        if not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
            raise ValueError("ranked task rationale must be non-empty")
    if seen != set(ABILITIES):
        raise ValueError("route must cover every catalog task exactly once")
    return result
