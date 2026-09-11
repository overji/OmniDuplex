#!/usr/bin/env python3
"""Common entry point for dataset generation tasks.

Example:
    python data_generation/generate.py --task event_alert \
        --video data_generation/videos/panorama.mp4 \
        data_generation/example.json -o data_generation/event_alert.json

    python data_generation/generate.py --task state_monitoring \
        --video data_generation/videos/panorama.mp4 \
        data_generation/example.json -o data_generation/state_monitoring.json

Use ``--print-prompt`` to inspect the sampled prompts without calling an API.
"""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
