"""Application inputs and generation results, independent of argparse."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(kw_only=True)
class APIOptions:
    provider: str
    model: str | None
    base_url: str
    api_key: str | None
    upload_url: str | None
    timeout: float
    retries: int


@dataclass(kw_only=True)
class GenerationOptions(APIOptions):
    task: str
    video: Path
    input: Path | None = None
    video_url: str | None = None
    usage_output: Path | None = None
    agent_sample_type: str | None = None


@dataclass(kw_only=True)
class BatchOptions(APIOptions):
    videos_dir: Path
    output_dir: Path
    tasks: list[str]
    samples_per_task: int
    seed: int
    resume: bool


@dataclass(kw_only=True)
class PipelineOptions(BatchOptions):
    video: Path | None = None
    split: str | None = None
    num_attempts: int | None = None
    video_url_map: Path | None = None
    agent_probability: float = 0.75
    agent_positive_probability: float = 1.0
    pr_probability: float = 0.5


@dataclass
class GenerationResult:
    sample: Any
    usage_record: dict[str, Any]
