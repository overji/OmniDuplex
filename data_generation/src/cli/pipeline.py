"""Video → caption → routing → 1–8 balanced abilities, using remote APIs."""
import argparse
import sys
from pathlib import Path

from .api_config import add_api_arguments, resolve_api_arguments
from .options import application_options
from ..application.options import PipelineOptions
from ..application import pipeline
from ..domain.catalog import TASK_CATALOG


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--video", type=Path, help="one video (legacy convenience)")
    sources.add_argument("--videos-dir", type=Path, help="recursively process a video directory once")
    parser.add_argument("--num-attempts", type=int, help="global QA attempt limit, including skip and failure")
    parser.add_argument("--video-url-map", type=Path, help="JSON map of video relative paths to remote URLs")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="historical output parent; with --resume, the specific run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pr-probability", type=float, default=0.5)
    parser.add_argument("--agent-probability", type=float, default=0.75)
    parser.add_argument("--agent-positive-probability", type=float, default=0.7)
    add_api_arguments(parser)
    args = resolve_api_arguments(parser.parse_args(argv))
    for name in ("pr_probability", "agent_probability", "agent_positive_probability"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.num_attempts is not None and args.num_attempts < 1:
        parser.error("--num-attempts must be positive")
    if args.videos_dir is not None and args.num_attempts is None:
        parser.error("directory pipeline requires --num-attempts")
    args.video = args.video.expanduser().resolve() if args.video else None
    args.output_dir = args.output_dir.expanduser().resolve()
    args.videos_dir = args.video.parent if args.video else args.videos_dir.expanduser().resolve()
    if args.video_url_map:
        args.video_url_map = args.video_url_map.expanduser().resolve()
    args.tasks = list(TASK_CATALOG)
    args.samples_per_task = 1
    return args


def main(argv=None):
    try:
        return pipeline.run(application_options(PipelineOptions, parse_args(argv)),
                            progress=lambda message, failed: print(message, file=sys.stderr if failed else sys.stdout, flush=True))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
