#!/usr/bin/env python3
"""Generate duration-aware video-captioning prompts from a video file.

Example:
    python data_generation/generate_video_prompts.py \
        data_generation/videos/panorama.mp4 -o data_generation/rendered_prompts
"""

from src.video_caption import main


if __name__ == "__main__":
    raise SystemExit(main())
