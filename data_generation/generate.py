#!/usr/bin/env python3
"""Common entry point for dataset generation tasks.

Example:
    python data_generation/generate.py --task event_alert \
        --video data_generation/videos/test.mp4 \
        path/to/events.json -o data_generation/event_alert.json

    python data_generation/generate.py --task target_grounding \
        --video data_generation/videos/test.mp4 \
        path/to/events.json --print-prompt

Use ``--print-prompt`` to inspect the sampled prompts without calling an API.
See data_generation/README.md for all thirteen tasks and the five backend workflows.
"""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
