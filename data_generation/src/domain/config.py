"""Configuration and randomized generation choices."""

from __future__ import annotations

import math
import random


AUDIO_PREFERENCE_PROBABILITY = 0.7
START_INPUT_PROBABILITY = 0.5
AGENT_DELAY_MIN_SECONDS = 5
AGENT_DELAY_MIN_FRACTION = 0.1
AGENT_DELAY_MAX_FRACTION = 0.5
AGENT_REFUSAL_WINDOW_SECONDS = 3
# Counts 1–8: 15%, 40%, 30%, 13%, then 0.5% each; expected count = 2.5.
PIPELINE_ABILITY_COUNT_WEIGHTS = (30, 80, 60, 26, 1, 1, 1, 1)


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
    lower = max(
        AGENT_DELAY_MIN_SECONDS,
        math.ceil(duration_seconds * AGENT_DELAY_MIN_FRACTION),
    )
    upper = math.floor(duration_seconds * AGENT_DELAY_MAX_FRACTION)
    if upper < lower:
        return None
    return random.randint(lower, upper)


AGENT_POSITIVE_PROBABILITY = 1.0


def choose_agent_sample_type() -> str:
    """New Agent data always has the capabilities required for delegation."""
    return "positive"


def validate_agent_positive_probability(value: float) -> None:
    if validate_probability("agent_positive_probability", value) != 1:
        raise ValueError("Agent negative generation is disabled; agent_positive_probability must be 1")
