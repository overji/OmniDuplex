"""Count agent generation."""
from pathlib import Path
from typing import Any

from . import _agent

GenerationContext = _agent.GenerationContext
GenerationRequest = _agent.GenerationRequest


def build_request(path: Path, video_path: Path) -> GenerationRequest:
    return _agent.build_request(path, video_path, "count_agent")


def validate_result(result: dict[str, Any], context: GenerationContext) -> dict[str, Any]:
    return _agent.validate_result(result, context, "count_agent")
