"""The minimal result contract between native adapters and the execution layer."""
from dataclasses import dataclass


@dataclass
class SampleResult:
    events: list[dict]
    observed_seconds: float
    status: str = "complete"
    include_sample_type: bool = True
