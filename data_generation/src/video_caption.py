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

from .common import format_mmss, timestamp_seconds


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "video_caption"
CAPTION_PROMPT_PATH = PROMPT_DIR / "caption.md"


@dataclass(frozen=True)
class VideoPromptBundle:
    video_path: Path
    duration_seconds: float
    duration: str
    caption_prompt: str


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


def validate_caption(value: Any, duration_seconds: float) -> list[dict[str, str]]:
    """Validate the paper's dense-caption array against the source duration."""
    if not isinstance(value, list) or not value:
        raise ValueError("caption must be a non-empty JSON array")
    end_of_video = timestamp_seconds(format_mmss(duration_seconds))
    if end_of_video == 0:
        raise ValueError("video is too short for MM:SS caption segments")
    previous_end = 0
    fields = {"start", "end", "caption", "visual", "audio", "speech"}
    for segment in value:
        if not isinstance(segment, dict) or set(segment) != fields:
            raise ValueError("caption segment requires start, end, caption, visual, audio, speech")
        for field in fields:
            if not isinstance(segment[field], str) or not segment[field].strip():
                raise ValueError(f"caption segment {field} must be non-empty text")
        start = timestamp_seconds(segment["start"])
        end = timestamp_seconds(segment["end"])
        if start != previous_end or not 0 < end - start <= 45:
            raise ValueError("caption segments must be contiguous, increasing, and at most 45 seconds")
        if end > end_of_video:
            raise ValueError("caption exceeds video duration")
        previous_end = end
    if previous_end != end_of_video:
        raise ValueError("caption must cover the entire video")
    return value


def load_caption(path: Path, duration_seconds: float) -> list[dict[str, str]]:
    return validate_caption(json.loads(path.read_text(encoding="utf-8")), duration_seconds)


def build_video_prompts(video_path: Path, *, ffprobe: str = "ffprobe") -> VideoPromptBundle:
    resolved = video_path.expanduser().resolve()
    seconds = probe_video_duration(resolved, ffprobe=ffprobe)
    duration = format_mmss(seconds)
    if timestamp_seconds(duration) == 0:
        raise ValueError("video is too short for MM:SS caption segments")
    prompt = CAPTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    for key, value in {
        "{duration_mmss}": duration,
        "{duration_sec:.0f}": str(timestamp_seconds(duration)),
        "{suggested_segments}": str(max(1, math.ceil(seconds / 15))),
    }.items():
        prompt = prompt.replace(key, value)
    return VideoPromptBundle(resolved, seconds, duration, prompt)


def _write_prompts(bundle: VideoPromptBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "caption.prompt.md").write_text(bundle.caption_prompt + "\n", encoding="utf-8")
    (output_dir / "video-input.md").write_text(
        f"[video_url: {bundle.video_path}]\n\nAttach the original video, including its audio. "
        "Rendering substitutes text; it does not inspect or verify the video.\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one dense audiovisual caption prompt.")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    bundle = build_video_prompts(args.video, ffprobe=args.ffprobe)
    if args.output_dir:
        _write_prompts(bundle, args.output_dir)
        print(f"caption prompt: {args.output_dir / 'caption.prompt.md'}")
    else:
        print(f"[video_url: {bundle.video_path}]\n{bundle.caption_prompt}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
