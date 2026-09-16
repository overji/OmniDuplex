"""Meeting understanding and follow-up task construction and validation."""

from pathlib import Path
from typing import Any

from . import _shared

PROMPT_PATH = _shared.PROMPT_ROOT / "agent" / "meeting_understanding.md"
GenerationContext = _shared.GenerationContext
GenerationRequest = _shared.GenerationRequest


def build_request(path: Path, video_path: Path, *, objects_path: Path | None = None) -> GenerationRequest:
    return _shared.build_request(
        path, video_path, PROMPT_PATH, objects_path=objects_path, agent=True
    )


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    validated = _shared.validate_agent_result(result, context)
    if validated.get("status") == "ok":
        envelope = _shared.agent_result_envelope(validated)
        if (
            envelope["status"] == "success"
            and not envelope["actions"]
            and not envelope["artifacts"]
        ):
            raise ValueError("successful meeting workflows require an action or artifact")
    return validated
