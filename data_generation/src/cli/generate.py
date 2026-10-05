"""Command-line entry point for one generated sample."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .api_config import add_api_arguments, resolve_api_arguments
from .options import application_options
from ..application.options import GenerationOptions
from ..application.generation import prepare_request, generate
from ..application.registry import task_names


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
    add_api_arguments(parser)
    parser.add_argument("--video-url", help="remote HTTP(S) or Qwen temporary OSS URL; Qwen uploads the local video automatically when omitted")
    parser.add_argument("--usage-output", type=Path, help="write provider token usage and estimated cost to this JSON sidecar")
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
    return resolve_api_arguments(args)


def run(args: argparse.Namespace) -> int:
    options = application_options(GenerationOptions, args)
    request = prepare_request(options)
    if args.print_prompt:
        if args.task == "caption":
            print(request.caption_prompt)
            print()
            print("Annotate the attached video.")
        else:
            print(request.system_prompt)
            print()
            print(request.user_prompt)
        return 0
    result = generate(options, request).sample
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
