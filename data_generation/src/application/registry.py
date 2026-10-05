"""Registry derived from the complete interaction matrix; route is auxiliary."""
from ..domain.catalog import TASK_CATALOG
from ..domain.tasks import route
from ..domain.tasks.interaction import InteractionTask

from ..infrastructure.paths import PROMPT_ROOT

TASKS = {name: InteractionTask(name) for name in TASK_CATALOG}
TASKS["route"] = route


def task_names() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def get_task(name: str):
    try:
        task = TASKS[name]
        if isinstance(task, InteractionTask) and task.mode == "agent":
            template = (PROMPT_ROOT / "task_generation" / "agent" / "capabilities.md").read_text(encoding="utf-8")
            return InteractionTask(name, template)
        return task
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}; available tasks: {', '.join(task_names())}") from exc
