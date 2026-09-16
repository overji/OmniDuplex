"""Configuration and randomized generation choices."""

from __future__ import annotations

import math
import random


AUDIO_PREFERENCE_PROBABILITY = 0.5
START_INPUT_PROBABILITY = 0.5
AGENT_DELAY_MIN_SECONDS = 15
AGENT_DELAY_MAX_SECONDS = 45


def validate_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def choose_preferred_modality(
    probability: float = AUDIO_PREFERENCE_PROBABILITY,
) -> str:
    probability = validate_probability("AUDIO_PREFERENCE_PROBABILITY", probability)
    return "audio" if random.random() < probability else "vision"


def choose_question_position(
    duration_seconds: float,
    probability: float = START_INPUT_PROBABILITY,
) -> str:
    probability = validate_probability("START_INPUT_PROBABILITY", probability)
    if duration_seconds < 1.0:
        return "start"
    return "start" if random.random() < probability else "non_start"


def choose_agent_delay(duration_seconds: float) -> int | None:
    """Choose an in-video simulated backend delay, or None when impossible."""
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds < 0
    ):
        raise ValueError(f"invalid video duration: {duration_seconds!r}")
    upper = min(AGENT_DELAY_MAX_SECONDS, math.floor(duration_seconds))
    if upper < AGENT_DELAY_MIN_SECONDS:
        return None
    return random.randint(AGENT_DELAY_MIN_SECONDS, upper)
