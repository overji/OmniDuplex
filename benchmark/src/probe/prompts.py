"""Build the whitelisted input boundary for a single independent probe."""
from dataclasses import dataclass
from pathlib import Path

from benchmark.data.prompts import build_system_prompt
from benchmark.probe.data import REQUEST_FIELDS


@dataclass(frozen=True)
class ProbeInput:
    system_prompt: str
    user_prompt: str
    video: Path
    has_audio: bool
    prefix_seconds: float


def model_input(row, video, template):
    request = {key: row['request'][key] for key in REQUEST_FIELDS if key in row['request']}
    system = build_system_prompt(template, request)
    user = (f"Original standing request: {request['question']}\n"
            f"Request issued at video time {request['question_time']}.\n"
            f"This video prefix ends at {row['prefix_end_seconds']:g} seconds from video start.\n"
            f"Earlier reply opportunities already handled: {row['handled_count']}.\n"
            "Has at least one ADDITIONAL opportunity to respond become warranted?\n"
            "Answer NO, or YES: followed by a brief explanation.")
    return ProbeInput(system, user, video, bool(row['has_audio']), row['prefix_end_seconds'])
