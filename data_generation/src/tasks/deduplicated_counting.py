"""Deduplicated semantic target counting construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "counting" / "deduplicated_counting.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path, *, objects_path: Path | None = None) -> GenerationRequest:
    return _shared.build_request(
        path, video_path, PROMPT_PATH, objects_path=objects_path, historical=True
    )


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    return _shared.validate_counting_history(result, context, deduplicated=True)
