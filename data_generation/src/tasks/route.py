"""Video-to-task routing prompt construction and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..video_caption import load_caption, probe_video_duration

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "route.md"
TASK_CATALOG: dict[str, str] = {
    "event_alert": "Alert on repeated or discrete verified events.",
    "target_grounding": "Report where a specified visual target or sound is located.",
    "snapshot_counting": "Count clearly distinguishable targets at trigger moments.",
    "semantic_condition_alert": "Alert on an abstract condition supported by context.",
    "event_narration": "Provide concise realtime updates about important changes.",
    "cumulative_counting": "Count independent occurrences of a specified event over time.",
    "deduplicated_counting": "Count distinct semantic targets while merging repeat appearances.",
    "state_monitoring": "Report transitions of one observable state.",
    "event_agent": "Handle transactions or complex reasoning after a specified event.",
    "condition_agent": "Prepare information, transactions or advice when all semantic conditions hold.",
    "count_agent": "Organize or adjust plans when a specified count crosses a threshold.",
    "target_agent": "Handle a related transaction or complex reasoning when a matching target appears.",
}


@dataclass(frozen=True)
class GenerationContext:
    duration_seconds: float


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    user_prompt: str
    context: GenerationContext
    video_path: Path


def build_request(path: Path, video_path: Path) -> GenerationRequest:
    duration = probe_video_duration(video_path.expanduser().resolve())
    evidence = {"caption": load_caption(path, duration)}
    catalog = "\n".join(f"- {name}: {description}" for name, description in TASK_CATALOG.items())
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{{VIDEO_DURATION}}", f"{duration:.0f} seconds")
    prompt = prompt.replace("{{TASK_CATALOG}}", catalog)
    if "{{" in prompt:
        raise ValueError("unresolved route prompt placeholder")
    return GenerationRequest(
        system_prompt=prompt,
        user_prompt=json.dumps(evidence, ensure_ascii=False, indent=2),
        context=GenerationContext(duration),
        video_path=video_path.expanduser().resolve(),
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
        if not isinstance(item, dict) or item.get("task") not in TASK_CATALOG:
            raise ValueError("invalid unsuitable task")
        task = item["task"]
        if task in seen:
            raise ValueError("duplicate routed task")
        seen.add(task)
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("unsuitable task reason must be non-empty")
    for expected_rank, item in enumerate(ranked, 1):
        if not isinstance(item, dict) or item.get("task") not in TASK_CATALOG:
            raise ValueError("invalid ranked task")
        task = item["task"]
        if task in seen:
            raise ValueError("duplicate routed task")
        seen.add(task)
        if item.get("rank") != expected_rank:
            raise ValueError("ranks must be consecutive")
        if not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
            raise ValueError("ranked task rationale must be non-empty")
    if seen != set(TASK_CATALOG):
        raise ValueError("route must cover every catalog task exactly once")
    return result
