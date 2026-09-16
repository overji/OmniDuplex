"""Event-driven realtime narration task construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "narration" / "event_narration.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path, *, objects_path: Path | None = None) -> GenerationRequest:
    return _shared.build_request(path, video_path, PROMPT_PATH, objects_path=objects_path)


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    validated, question_seconds = _shared.validate_common(
        result, context, min_responses=3, max_responses=6, max_words=40
    )
    if question_seconds is None:
        return validated
    for index, response in enumerate(result["responses"]):
        facts = response.get("observed_facts")
        if not isinstance(facts, list) or not facts or not all(
            isinstance(fact, str) and fact.strip() for fact in facts
        ):
            raise ValueError(f"responses[{index}].observed_facts must be non-empty")
        interpretation = response.get("interpretation")
        if interpretation is not None and not isinstance(interpretation, str):
            raise ValueError("interpretation must be a string or null")
    return result
