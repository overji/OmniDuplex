"""Registry of supported data-generation tasks."""

from __future__ import annotations

from types import ModuleType

from .tasks import (
    cumulative_counting,
    deduplicated_counting,
    event_alert,
    event_narration,
    semantic_condition_alert,
    snapshot_counting,
    state_monitoring,
    target_grounding,
    triggered_knowledge_retrieval,
    triggered_realtime_query,
    triggered_multi_source_synthesis,
    triggered_content_recording,
    triggered_message_notification,
    route,
)


TASKS: dict[str, ModuleType] = {
    "cumulative_counting": cumulative_counting,
    "deduplicated_counting": deduplicated_counting,
    "event_alert": event_alert,
    "event_narration": event_narration,
    "semantic_condition_alert": semantic_condition_alert,
    "snapshot_counting": snapshot_counting,
    "state_monitoring": state_monitoring,
    "target_grounding": target_grounding,
    "triggered_knowledge_retrieval": triggered_knowledge_retrieval,
    "triggered_realtime_query": triggered_realtime_query,
    "triggered_multi_source_synthesis": triggered_multi_source_synthesis,
    "triggered_content_recording": triggered_content_recording,
    "triggered_message_notification": triggered_message_notification,
    "route": route,
}


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def get_task(name: str) -> ModuleType:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ", ".join(task_names())
        raise ValueError(f"unknown task {name!r}; available tasks: {choices}") from exc
