"""Validate stored samples and combine them into QA bundles."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..domain.common import SAMPLE_SCHEMA
from ..domain.qa import adapt_responses
from .registry import TASKS, get_task


def video_name(folder: Path, videos_dir: Path) -> str:
    candidates = [folder.name]
    if folder.name.endswith("_video"):
        candidates.append(folder.name.removesuffix("_video"))
    for stem in candidates:
        candidate = videos_dir / f"{stem}.mp4"
        if candidate.is_file():
            return candidate.name
    raise ValueError(f"no video found for {folder.name} in {videos_dir}")


def load_sample(source: Path, videos_dir: Path) -> dict[str, Any] | None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SAMPLE_SCHEMA:
        raise ValueError(f"unsupported sample schema: {source}")
    name = data.get("task")
    if name not in TASKS or name == "route":
        raise ValueError(f"unsupported sample task: {source}")
    task = get_task(name)
    raw_context = data.get("generation_context")
    if not isinstance(raw_context, dict):
        raise ValueError(f"missing generation_context: {source}")
    try:
        context = task.GenerationContext(**raw_context)
    except TypeError as exc:
        raise ValueError(f"invalid generation_context: {source}") from exc
    data = task.validate_result(data, context, allow_legacy_agent_windows=True)
    if data.get("status") == "skip":
        return None
    result = dict(data)
    result["responses"] = adapt_responses(data["responses"])
    folder = source.parent.parent if source.parent.name == "qa" else source.parent
    result["video"] = data.get("video") or video_name(folder, videos_dir)
    result["task"] = name
    result["video_folder"] = folder.name
    return result


def build_bundle(data_dir: Path, videos_dir: Path) -> list[dict[str, Any]]:
    bundle = []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
        sample_dir = folder / "qa" if (folder / "qa").is_dir() else folder
        for source in sorted(sample_dir.glob("*.json")):
            if source.stem in {"caption", "route", "all_qa", "coverage", "manifest"} or source.name.endswith(".usage.json"):
                continue
            sample = load_sample(source, videos_dir)
            if sample is not None:
                bundle.append(sample)
    return bundle


