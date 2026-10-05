"""Prepare video-caption requests and export prompt files."""
import json
from pathlib import Path
from ..domain import caption
from ..infrastructure.video import probe_video_duration
from ..infrastructure.paths import PROMPT_ROOT


def load_caption(path: Path, duration_seconds: float):
    return caption.validate_caption(json.loads(path.read_text(encoding="utf-8")), duration_seconds)


def build_video_prompts(video_path: Path, *, ffprobe: str = "ffprobe"):
    resolved = video_path.expanduser().resolve()
    seconds = probe_video_duration(resolved, ffprobe=ffprobe)
    prompt = (PROMPT_ROOT / "video_caption" / "caption.md").read_text(encoding="utf-8")
    return caption.build_video_prompts(resolved, seconds, prompt)


def write_prompts(bundle: caption.VideoPromptBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "caption.prompt.md").write_text(bundle.caption_prompt + "\n", encoding="utf-8")
    (output_dir / "video-input.md").write_text(
        "Attach the original video, including its audio. "
        "Rendering substitutes text; it does not inspect or verify the video.\n",
        encoding="utf-8",
    )


