"""Common JSON and timestamp helpers for data-generation tasks."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


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


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = candidate.find("{")
        if start < 0:
            raise ValueError("model response contains no JSON object")
        try:
            result, end = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned invalid JSON: {exc}") from exc
        if candidate[start + end :].strip():
            raise ValueError("model response contains text after the JSON object")
    if not isinstance(result, dict):
        raise ValueError("model response must be a JSON object")
    return result


SAMPLE_SCHEMA = "omni-interaction-v2"


def extract_json_array(text: str) -> list:
    """Caption responses use an array, never the task object's extraction fallback."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    value = json.loads(candidate)
    if not isinstance(value, list):
        raise ValueError("caption response must be a JSON array")
    return value
