"""Target grounding task construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "proactive_alert" / "target_grounding.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path, *, objects_path: Path | None = None) -> GenerationRequest:
    return _shared.build_request(path, video_path, PROMPT_PATH, objects_path=objects_path)


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    validated, question_seconds = _shared.validate_common(
        result, context, min_responses=1, max_responses=4, max_words=20
    )
    if question_seconds is None:
        return validated
    for index, response in enumerate(result["responses"]):
        _shared.require_text(response.get("target"), f"responses[{index}].target")
        location_type = response.get("location_type")
        if location_type not in {"visual", "sound"}:
            raise ValueError("location_type must be visual or sound")
        _shared.require_text(response.get("location"), f"responses[{index}].location")
        if location_type == "visual":
            region = response.get("frame_region")
            rows = {"top", "middle", "bottom"}
            columns = {"left", "center", "right"}
            if not isinstance(region, str) or "-" not in region:
                raise ValueError("visual location requires a 3x3 frame_region")
            row, column = region.split("-", 1)
            if row not in rows or column not in columns:
                raise ValueError("invalid frame_region")
            if result["vision_dependency"] != "required":
                raise ValueError("visual grounding requires vision")
        elif result["audio_dependency"] != "required":
            raise ValueError("sound grounding requires audio")
        confidence = response.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    return result
