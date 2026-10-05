"""Foreground, sequential generation with validated content-addressed resume."""
import argparse
import sys
from pathlib import Path
from .api_config import add_api_arguments, resolve_api_arguments
from .options import application_options
from ..domain.catalog import TASK_CATALOG
from ..application.options import BatchOptions
from ..application import batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="parent directory for a new timestamped run; with --resume, the specific run directory")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_CATALOG), default=list(TASK_CATALOG))
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    add_api_arguments(parser)
    args = parser.parse_args(argv)
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be positive")
    args.tasks = list(dict.fromkeys(args.tasks))
    return resolve_api_arguments(args)


def run(args):
    def progress(message, failed):
        print(message, file=sys.stderr if failed else sys.stdout, flush=True)

    return batch.run(application_options(BatchOptions, args), progress=progress)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
