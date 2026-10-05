"""Render one dense audiovisual caption prompt."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from ..application.video_caption import build_video_prompts, write_prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one dense audiovisual caption prompt.")
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    bundle = build_video_prompts(args.video, ffprobe=args.ffprobe)
    if args.output_dir:
        write_prompts(bundle, args.output_dir)
        print(f"caption prompt: {args.output_dir / 'caption.prompt.md'}")
    else:
        print(bundle.caption_prompt)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
