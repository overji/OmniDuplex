"""Registry of supported data-generation tasks."""

from __future__ import annotations

from types import ModuleType

from .tasks import (
    cumulative_counting,
    deduplicated_counting,
    event_alert,
    event_narration,
    meeting_understanding,
    semantic_condition_alert,
    snapshot_counting,
    state_monitoring,
    target_grounding,
    triggered_information_query,
    triggered_tool_action,
)


TASKS: dict[str, ModuleType] = {
    "cumulative_counting": cumulative_counting,
    "deduplicated_counting": deduplicated_counting,
    "event_alert": event_alert,
    "event_narration": event_narration,
    "meeting_understanding": meeting_understanding,
    "semantic_condition_alert": semantic_condition_alert,
    "snapshot_counting": snapshot_counting,
    "state_monitoring": state_monitoring,
    "target_grounding": target_grounding,
    "triggered_information_query": triggered_information_query,
    "triggered_tool_action": triggered_tool_action,
}


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def get_task(name: str) -> ModuleType:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ", ".join(task_names())
        raise ValueError(f"unknown task {name!r}; available tasks: {choices}") from exc
