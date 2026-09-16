"""Triggered snapshot counting task construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "counting" / "snapshot_counting.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path, *, objects_path: Path | None = None) -> GenerationRequest:
    return _shared.build_request(path, video_path, PROMPT_PATH, objects_path=objects_path)


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    validated, question_seconds = _shared.validate_common(
        result, context, min_responses=1, max_words=20
    )
    if question_seconds is None:
        return validated
    if result["vision_dependency"] != "required":
        raise ValueError("snapshot counting requires vision")
    for index, response in enumerate(result["responses"]):
        _shared.require_count(response.get("count"), f"responses[{index}].count")
        _shared.require_text(
            response.get("counting_basis"), f"responses[{index}].counting_basis"
        )
    return result
