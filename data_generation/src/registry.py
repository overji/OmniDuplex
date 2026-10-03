"""Registry derived from the complete interaction matrix; route is auxiliary."""
from .catalog import TASK_CATALOG
from .tasks import route
from .tasks.interaction import InteractionTask

TASKS = {name: InteractionTask(name) for name in TASK_CATALOG}
TASKS["route"] = route


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def get_task(name: str):
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}; available tasks: {', '.join(task_names())}") from exc
