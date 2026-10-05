"""Load request inputs before invoking pure task builders."""
from ..infrastructure.paths import PROMPT_ROOT
from ..infrastructure.video import probe_video_duration
from .video_caption import load_caption
from .registry import get_task


def build_task_request(name, path, video_path):
    video = video_path.expanduser().resolve()
    duration = probe_video_duration(video)
    caption = load_caption(path, duration)
    task = get_task(name)
    if name == "route":
        prompt = (PROMPT_ROOT / "route.md").read_text(encoding="utf-8")
        return task.build_request(video, duration, caption, prompt)
    root = PROMPT_ROOT / "task_generation"
    mode_path = root / ("agent/generation.md" if task.mode == "agent" else f"{task.mode}.md")
    return task.build_request(video, duration, caption,
                              (root / "common.md").read_text(), mode_path.read_text())
