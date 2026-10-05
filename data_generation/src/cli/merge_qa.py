"""Combine per-video task results into a uniform QA bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ..infrastructure.paths import ROOT
from ..application.merge_qa import build_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "data" / "all_qa.json")
    args = parser.parse_args(argv)
    try:
        bundle = build_bundle(args.data_dir, args.videos_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
