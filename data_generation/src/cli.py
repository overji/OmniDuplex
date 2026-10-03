"""Common command-line entry point for dataset generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .client import chat_completion
from .common import SAMPLE_SCHEMA, extract_json, extract_json_array
from .registry import get_task, task_names
from .video_caption import build_video_prompts, validate_caption


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one dataset sample.")
    parser.add_argument("--task", required=True, choices=(*task_names(), "caption"))
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="local source video sent to the model",
    )
    parser.add_argument("input", nargs="?", type=Path, help="caption.json (required except for caption)")
    parser.add_argument("-o", "--output", type=Path, help="output path (default: stdout)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"), help="model name")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="print sampled system/user prompts without calling the API",
    )
    args = parser.parse_args(argv)
    if args.task == "caption" and args.input is not None:
        parser.error("caption accepts the video only, not an input JSON")
    if args.task != "caption" and args.input is None:
        parser.error("task generation requires caption.json")
    return args


def prepare_request(args):
    if args.task == "caption":
        return build_video_prompts(args.video)
    return get_task(args.task).build_request(args.input, args.video)


def generate(args: argparse.Namespace, request=None):
    request = request if request is not None else prepare_request(args)
    if args.task == "caption":
        bundle = request
        system_prompt, user_prompt, video_path = bundle.caption_prompt, "Annotate the attached video.", bundle.video_path
    else:
        task = get_task(args.task)
        system_prompt, user_prompt, video_path = request.system_prompt, request.user_prompt, request.video_path
    if args.print_prompt:
        print(f"[video_url: {video_path}]")
        print("\n---\n")
        print(system_prompt)
        print("\n---\n")
        print(user_prompt)
        return None
    if not args.api_key:
        raise ValueError("missing API key: pass --api-key or set OPENAI_API_KEY")
    if not args.model:
        raise ValueError("missing model: pass --model or set OPENAI_MODEL")

    raw_response = chat_completion(
        system_prompt,
        user_prompt,
        video_path=video_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=max(0, args.retries),
        json_array=args.task == "caption",
    )
    if args.task == "caption":
        result = validate_caption(extract_json_array(raw_response), bundle.duration_seconds)
    else:
        result = task.validate_result(extract_json(raw_response), request.context)
        result.update({
            "schema": SAMPLE_SCHEMA,
            "task": args.task,
            "video": video_path.name,
            "generation_context": asdict(request.context),
        })
    return result


def run(args: argparse.Namespace) -> int:
    result = generate(args)
    if result is None:
        return 0
    output_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        sys.stdout.write(output_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
