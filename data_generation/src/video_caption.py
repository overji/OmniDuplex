"""Build duration-aware prompts for the video-captioning pipeline."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "video_caption"
EVENTS_PROMPT_PATH = PROMPT_DIR / "events.md"
OBJECTS_PROMPT_PATH = PROMPT_DIR / "objects.md"
_DURATION_PLACEHOLDER = "{{VIDEO_DURATION}}"
_SECONDS_PLACEHOLDER = "{{VIDEO_DURATION_SECONDS}}"


@dataclass(frozen=True)
class VideoPromptBundle:
    """Rendered prompts and the probed duration of their source video."""

    video_path: Path
    duration_seconds: float
    duration: str
    events_prompt: str
    objects_prompt: str


def _positive_finite_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def probe_video_duration(video_path: Path, *, ffprobe: str = "ffprobe") -> float:
    """Return the video duration in seconds using ffprobe metadata."""
    path = video_path.expanduser()
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffprobe executable not found: {ffprobe!r}; install FFmpeg or pass "
            "--ffprobe"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"could not read video metadata: {detail}") from exc

    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    format_metadata = metadata.get("format", {})
    duration = _positive_finite_duration(format_metadata.get("duration"))
    if duration is not None:
        return duration

    stream_durations = [
        parsed
        for stream in metadata.get("streams", [])
        if isinstance(stream, dict)
        for parsed in [_positive_finite_duration(stream.get("duration"))]
        if parsed is not None
    ]
    if stream_durations:
        return max(stream_durations)
    raise RuntimeError("ffprobe metadata does not contain a valid video duration")


def video_to_data_url(video_path: Path) -> str:
    """Encode a local video as a data URL suitable for a video_url content block."""
    path = video_path.expanduser()
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("video/"):
        raise ValueError(f"cannot determine video MIME type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def format_video_duration(duration_seconds: float) -> str:
    """Format a duration as HH:MM:SS.mmm, rounded to the nearest millisecond."""
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds < 0
    ):
        raise ValueError(f"invalid video duration: {duration_seconds!r}")
    total_milliseconds = int(float(duration_seconds) * 1000 + 0.5)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _render_prompt(
    template_path: Path,
    duration: str,
    seconds: float,
    *,
    replacements: dict[str, str] | None = None,
) -> str:
    template = template_path.read_text(encoding="utf-8").strip()
    for placeholder in (_DURATION_PLACEHOLDER, _SECONDS_PLACEHOLDER):
        if placeholder not in template:
            raise ValueError(
                f"missing prompt placeholder {placeholder} in {template_path}"
            )
    rendered = template.replace(_DURATION_PLACEHOLDER, duration).replace(
        _SECONDS_PLACEHOLDER,
        f"{seconds:.3f}",
    )
    for placeholder, value in (replacements or {}).items():
        if placeholder not in rendered:
            raise ValueError(f"missing prompt placeholder {placeholder} in {template_path}")
        rendered = rendered.replace(placeholder, value)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError(f"unresolved prompt placeholder in {template_path}")
    return rendered


def build_video_prompts(
    video_path: Path,
    *,
    ffprobe: str = "ffprobe",
    events_json: Path | None = None,
) -> VideoPromptBundle:
    """Probe a video and render the events and objects prompts for it."""
    resolved_path = video_path.expanduser().resolve()
    duration_seconds = probe_video_duration(resolved_path, ffprobe=ffprobe)
    duration = format_video_duration(duration_seconds)
    required_fields = ("video_summary", "audio_events", "visual_events")
    event_replacements = {
        f"{{{{{field.upper()}}}}}": "{}" if field == "video_summary" else "[]"
        for field in required_fields
    }
    event_replacements["{{EVENT_UNCERTAINTIES}}"] = "[]"
    if events_json is not None:
        metadata_path = events_json.expanduser()
        if not metadata_path.is_file():
            raise ValueError(f"events JSON file does not exist: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid events JSON: {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("events JSON must contain a top-level object")
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            raise ValueError(
                "events JSON is missing required fields: " + ", ".join(missing_fields)
            )
        event_replacements = {
            f"{{{{{field.upper()}}}}}": json.dumps(
                metadata[field], ensure_ascii=False, indent=2
            )
            for field in required_fields
        }
        event_replacements["{{EVENT_UNCERTAINTIES}}"] = json.dumps(
            metadata.get("uncertainties", []), ensure_ascii=False, indent=2
        )
    return VideoPromptBundle(
        video_path=resolved_path,
        duration_seconds=duration_seconds,
        duration=duration,
        events_prompt=_render_prompt(
            EVENTS_PROMPT_PATH,
            duration,
            duration_seconds,
        ),
        objects_prompt=_render_prompt(
            OBJECTS_PROMPT_PATH,
            duration,
            duration_seconds,
            replacements=event_replacements,
        ),
    )


def _write_prompts(bundle: VideoPromptBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "events.prompt.md").write_text(
        bundle.events_prompt + "\n",
        encoding="utf-8",
    )
    (output_dir / "objects.prompt.md").write_text(
        bundle.objects_prompt + "\n",
        encoding="utf-8",
    )
    (output_dir / "video-input.md").write_text(
        f"[video_url: {bundle.video_path}]\n\n"
        "Attach this original video when using either caption prompt. "
        "For objects, supply the video together with the rendered events references. "
        "Rendering only substitutes text; it does not inspect or verify the video.\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate duration-aware events and objects prompts for a video."
    )
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="write events.prompt.md and objects.prompt.md to this directory",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable name or path",
    )
    parser.add_argument(
        "--events-json",
        type=Path,
        help="structured video events JSON to inject into the objects prompt",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    bundle = build_video_prompts(
        args.video,
        ffprobe=args.ffprobe,
        events_json=args.events_json,
    )
    if args.output_dir:
        _write_prompts(bundle, args.output_dir)
        print(f"video duration: {bundle.duration} ({bundle.duration_seconds:.3f}s)")
        print(f"events prompt: {args.output_dir / 'events.prompt.md'}")
        print(f"objects prompt: {args.output_dir / 'objects.prompt.md'}")
        print(f"video attachment instructions: {args.output_dir / 'video-input.md'}")
    else:
        print(f"[video_url: {bundle.video_path}]")
        print("--- events prompt ---")
        print(bundle.events_prompt)
        print("\n--- objects prompt ---")
        print(bundle.objects_prompt)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
