"""Dense-caption prompt rendering and validation without I/O."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .common import format_mmss, timestamp_seconds


@dataclass(frozen=True)
class VideoPromptBundle:
    video_path: Path
    duration_seconds: float
    duration: str
    caption_prompt: str


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


def build_video_prompts(video_path: Path, seconds: float, prompt: str) -> VideoPromptBundle:
    duration = format_mmss(seconds)
    if timestamp_seconds(duration) == 0:
        raise ValueError("video is too short for MM:SS caption segments")
    prompt = prompt.strip()
    for key, value in {
        "{duration_mmss}": duration,
        "{duration_sec:.0f}": str(timestamp_seconds(duration)),
        "{suggested_segments}": str(max(1, math.ceil(seconds / 15))),
    }.items():
        prompt = prompt.replace(key, value)
    return VideoPromptBundle(video_path, seconds, duration, prompt)


