"""Read video duration using ffprobe."""
from __future__ import annotations
import math
import json
import subprocess
from pathlib import Path
from typing import Any


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


