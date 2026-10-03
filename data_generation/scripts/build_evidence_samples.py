"""Build video-prefix samples for response evidence-sufficiency evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "evidence_sufficiency.md"


def timestamp_seconds(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"timestamp must be MM:SS: {value!r}")
    minutes, seconds = (int(part) for part in parts)
    if seconds >= 60:
        raise ValueError(f"invalid timestamp: {value!r}")
    return minutes * 60 + seconds


def format_mmss(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_sample(source_json: Path, video: Path, output_dir: Path) -> None:
    source = load_object(source_json)
    question = source.get("question")
    responses = source.get("responses")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("source JSON must contain a question")
    if not isinstance(responses, list) or not responses:
        raise ValueError("source JSON must contain a non-empty responses array")

    if source.get("sample_type") in {"no_context", "missing_tool"}:
        raise ValueError("capability refusals are not audiovisual evidence samples")
    if any(isinstance(response, dict) and (
        "messages" in response or "role" in response or "tool_calls" in response
    ) for response in responses):
        raise ValueError("agent messages and simulated results are not audiovisual evidence samples")

    if source.get("ability") in {"prospective_reasoning", "causal_reasoning", "emotion_recognition"} or any(
        isinstance(response, dict) and isinstance(response.get("details"), dict)
        and response["details"].get("claim_type") in {"prediction", "inference"}
        for response in responses
    ):
        raise ValueError("predictions and inferences are not observed audiovisual facts")

    template = PROMPT_PATH.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_responses = []
    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"responses[{index}] must be an object")
        response_time = response.get("trigger_time")
        response_text = response.get("response")
        if not isinstance(response_time, str) or not isinstance(response_text, str):
            raise ValueError(f"responses[{index}] needs trigger_time and response")
        end_seconds = timestamp_seconds(response_time)
        clip_duration = end_seconds + 1
        clip_name = f"{video.stem}_response_{index:03d}_{response_time.replace(':', '-')}.mp4"
        clip_path = output_dir / clip_name
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(video),
                "-t", str(clip_duration), "-map", "0", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-c:a", "aac", str(clip_path),
            ],
            check=True,
        )
        prompt = template
        for placeholder, value in {
            "{{RESPONSE_TIME}}": response_time,
            "{{RESPONSE_INDEX}}": str(index),
            "{{QUESTION}}": question,
            "{{RESPONSE}}": response_text,
        }.items():
            prompt = prompt.replace(placeholder, value)
        prompt_path = output_dir / f"{clip_path.stem}.prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        expected = {
            "evidence_sufficient": True,
        }
        expected_path = output_dir / f"{clip_path.stem}.expected.json"
        write_json(expected_path, expected)
        manifest_responses.append({
            "response_index": index,
            "response_time": response_time,
            "response": response_text,
            "video": clip_name,
            "prompt": prompt_path.name,
            "expected": expected_path.name,
            "clip_start": "00:00",
            "clip_end_exclusive": format_mmss(clip_duration),
        })

    write_json(output_dir / "manifest.json", {
        "source_json": str(source_json),
        "source_video": str(video),
        "question": question,
        "responses": manifest_responses,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_json", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    build_sample(args.source_json, args.video, args.output_dir)


if __name__ == "__main__":
    main()
