"""Remove historical Agent negatives from source samples and QA bundles.

Pass source-data directories/files only; historical experiment snapshots should
retain their original sample order and input hashes. Defaults to a dry run.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation.src.domain.qa import is_agent_negative


def clean(paths, *, apply=False, backup_dir=None):
    if apply and backup_dir is None:
        raise ValueError("--apply requires --backup-dir")
    files = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files.update([path] if path.is_file() else path.rglob("*.json"))
    changes = []
    for path in sorted(files):
        if ".state" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            kept = [sample for sample in data if not is_agent_negative(sample)]
            removed = len(data) - len(kept)
        else:
            removed = int(is_agent_negative(data))
            kept = None
        if not removed:
            continue
        changes.append({"file": str(path), "removed": removed,
                        "remaining": len(kept) if kept is not None else 0})
        if apply:
            backup = backup_dir / path.resolve().relative_to(path.resolve().anchor)
            backup.parent.mkdir(parents=True, exist_ok=True)
            # Never overwrite the original backup on a repeated cleanup.
            if not backup.exists():
                shutil.copy2(path, backup)
            if kept is None:
                path.unlink()
            else:
                path.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(clean(args.paths, apply=args.apply, backup_dir=args.backup_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
