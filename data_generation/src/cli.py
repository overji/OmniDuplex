"""Common command-line entry point for dataset generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .client import chat_completion
from .common import extract_json
from .registry import get_task, task_names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one dataset sample.")
    parser.add_argument("--task", required=True, choices=task_names())
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="local source video sent to the model",
    )
    parser.add_argument("input", type=Path, help="task input JSON")
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
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    request = task.build_request(args.input, args.video)
    if args.print_prompt:
        print(f"[video_url: {request.video_path}]")
        print("\n---\n")
        print(request.system_prompt)
        print("\n---\n")
        print(request.user_prompt)
        return 0
    if not args.api_key:
        raise ValueError("missing API key: pass --api-key or set OPENAI_API_KEY")
    if not args.model:
        raise ValueError("missing model: pass --model or set OPENAI_MODEL")

    raw_response = chat_completion(
        request.system_prompt,
        request.user_prompt,
        video_path=request.video_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=max(0, args.retries),
    )
    result = task.validate_result(extract_json(raw_response), request.context)
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
