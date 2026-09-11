"""Registry of supported data-generation tasks."""

from __future__ import annotations

from types import ModuleType

from .tasks import event_alert, state_monitoring


TASKS: dict[str, ModuleType] = {
    "event_alert": event_alert,
    "state_monitoring": state_monitoring,
}


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def get_task(name: str) -> ModuleType:
    try:
        return TASKS[name]
    except KeyError as exc:
        choices = ", ".join(task_names())
        raise ValueError(f"unknown task {name!r}; available tasks: {choices}") from exc
