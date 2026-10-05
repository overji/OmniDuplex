"""JSON file and model-response adapters."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


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
