"""Combine per-video task results into a uniform QA bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import timestamp_seconds


ROOT = Path(__file__).resolve().parents[1]


def adapt_responses(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = []
    for response in responses:
        messages = response.get("messages")
        if messages is None:
            adapted.append(response)
            continue
        if not isinstance(messages, list) or not messages:
            raise ValueError("agent response must contain messages")
        parent_fields = {
            key: value for key, value in response.items()
            if key not in {"trigger_time", "messages"}
        }
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("agent message must be an object")
            timestamp = message.get("timestamp")
            timestamp_seconds(timestamp)
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("agent message content must be a string")
            entry = {"trigger_time": timestamp}
            if index == 0:
                entry["source_trigger_time"] = response.get("trigger_time")
                entry.update(parent_fields)
            entry["response"] = content
            entry.update({
                key: value for key, value in message.items()
                if key not in {"timestamp", "content"}
            })
            adapted.append(entry)
    return adapted


def video_name(folder: Path, videos_dir: Path) -> str:
    candidates = [folder.name]
    if folder.name.endswith("_video"):
        candidates.append(folder.name.removesuffix("_video"))
    for stem in candidates:
        candidate = videos_dir / f"{stem}.mp4"
        if candidate.is_file():
            return candidate.name
    raise ValueError(f"no video found for {folder.name} in {videos_dir}")


def build_bundle(data_dir: Path, videos_dir: Path) -> list[dict[str, Any]]:
    bundle = []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for source in sorted(folder.glob("*.json")):
            data = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("status") != "ok":
                continue
            if not isinstance(data.get("question"), str) or not isinstance(
                data.get("responses"), list
            ):
                continue
            result = dict(data)
            result["responses"] = adapt_responses(data["responses"])
            result["video"] = data.get("video") or video_name(folder, videos_dir)
            result["task"] = source.stem
            result["video_folder"] = folder.name
            bundle.append(result)
    return bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "all_qa.json")
    args = parser.parse_args(argv)
    try:
        bundle = build_bundle(args.data_dir, args.videos_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
