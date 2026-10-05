"""Pure timestamp helpers and sample schema."""
from __future__ import annotations
import math
import re
from typing import Any


def parse_duration(value: Any) -> float:
    """Parse HH:MM:SS(.sss), MM:SS(.sss), or a numeric second count."""
    if isinstance(value, bool):
        raise ValueError(f"invalid duration: {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("duration cannot be negative")
        if not math.isfinite(value):
            raise ValueError(f"invalid duration: {value!r}")
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing video duration")

    parts = value.strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid duration: {value!r}") from exc
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        if minutes >= 60:
            raise ValueError(f"invalid duration: {value!r}")
    elif len(numbers) == 2:
        hours, minutes, seconds = 0.0, numbers[0], numbers[1]
    else:
        raise ValueError(f"invalid duration: {value!r}")
    if (
        not all(math.isfinite(number) for number in numbers)
        or hours < 0
        or minutes < 0
        or seconds < 0
        or seconds >= 60
    ):
        raise ValueError(f"invalid duration: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_mmss(seconds: float) -> str:
    whole_seconds = int(seconds + 0.5)
    return f"{whole_seconds // 60:02d}:{whole_seconds % 60:02d}"


def timestamp_seconds(value: Any) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d{2,}:\d{2}", value):
        raise ValueError(f"timestamp must be MM:SS, got {value!r}")
    minutes, seconds = (int(part) for part in value.split(":"))
    if seconds >= 60:
        raise ValueError(f"timestamp has invalid seconds: {value!r}")
    return minutes * 60 + seconds


SAMPLE_SCHEMA = "omni-interaction-v2"
