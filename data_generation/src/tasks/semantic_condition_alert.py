"""Semantic condition alert task construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "proactive_alert" / "semantic_condition_alert.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path) -> GenerationRequest:
    return _shared.build_request(path, video_path, PROMPT_PATH)


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    validated, question_seconds = _shared.validate_common(
        result, context, min_responses=1, max_words=25
    )
    if question_seconds is None:
        return validated
    _shared.require_text(result.get("semantic_condition"), "semantic_condition")
    criteria = result.get("observable_criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("observable_criteria must be a non-empty string array")
    for criterion in criteria:
        _shared.require_text(criterion, "observable_criteria entry")
    for index, response in enumerate(result["responses"]):
        _shared.require_text(
            response.get("condition_evidence"),
            f"responses[{index}].condition_evidence",
        )
        confidence = response.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    return result
