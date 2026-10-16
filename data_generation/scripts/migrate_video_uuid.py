"""Rename the shared video library and update active QA references in place.

Run from the repository root. The original filename remains in video_metadata.json,
so an interrupted migration can be rerun and the mapping remains inspectable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[2]
VIDEOS = (ROOT / "videos").resolve()
METADATA = VIDEOS / "video_metadata.json"
DATA = ROOT / "data_generation/data"
BENCHMARK_DATA = ROOT / "benchmark/all_qa.json"
BENCHMARK_MANIFEST = ROOT / "benchmark/configs/manifests/data_manifest.json"
CACHE = ROOT / "benchmark/cache"
NAMESPACE = "OmniInteractionDataset/videos/"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".uuid-migration.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mapping_from_metadata(metadata: dict) -> dict[str, str]:
    mapping = {}
    for entry in metadata["videos"]:
        original = entry.get("original_file", entry["file"])
        expected = f"{uuid.uuid5(uuid.NAMESPACE_URL, NAMESPACE + original)}{Path(original).suffix.lower()}"
        if entry["file"] not in (original, expected):
            raise ValueError(f"Unexpected filename in metadata: {entry['file']}")
        mapping[original] = expected
    if len(mapping) != len(metadata["videos"]) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Metadata has duplicate source or target video names")
    actual = {path.name for path in VIDEOS.iterdir() if path.suffix.lower() == ".mp4"}
    for original, target in mapping.items():
        if (original in actual) == (target in actual):
            raise ValueError(f"Expected exactly one source or target video: {original}")
    if actual != (set(mapping) | set(mapping.values())) & actual:
        raise ValueError("Video directory contains files absent from metadata")
    return mapping


def migrate_videos(metadata: dict, mapping: dict[str, str]) -> None:
    for original, target in mapping.items():
        source = VIDEOS / original
        if source.exists():
            source.rename(VIDEOS / target)
    for entry in metadata["videos"]:
        original = entry.get("original_file", entry["file"])
        entry["original_file"] = original
        entry["file"] = mapping[original]
    metadata["schema_version"] = 2
    metadata["filename_rule"] = f"UUIDv5(URL namespace, {NAMESPACE}<original_file>)"
    metadata["description"] = "Metadata for all video files in this directory; original_file preserves the pre-migration filename."
    write_json(METADATA, metadata)


def migrate_json_references(mapping: dict[str, str]) -> int:
    replacements = sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True)
    changed = 0
    for path in [*DATA.rglob("*.json"), *DATA.rglob("*.jsonl"), BENCHMARK_DATA]:
        source = path.read_text(encoding="utf-8")
        updated = source
        for original, target in replacements:
            updated = updated.replace(original, target)
        if updated != source:
            # Preserve original JSON formatting and every field unrelated to video paths.
            if path.suffix == ".json":
                json.loads(updated)
            else:
                for line in updated.splitlines():
                    if line.strip():
                        json.loads(line)
            temporary = path.with_name(path.name + ".uuid-migration.tmp")
            temporary.write_text(updated, encoding="utf-8")
            os.replace(temporary, path)
            changed += 1
    return changed


def migrate_cache(mapping: dict[str, str]) -> int:
    changed = 0
    for original, target in mapping.items():
        old_stem, new_stem = Path(original).stem, Path(target).stem
        old_dir, new_dir = CACHE / old_stem, CACHE / new_stem
        if old_dir.exists():
            if new_dir.exists():
                raise ValueError(f"Both cache directories exist: {old_dir}, {new_dir}")
            old_dir.rename(new_dir)
            changed += 1
        if new_dir.exists():
            path = new_dir / "metadata.json"
            cache = json.loads(path.read_text())
            cache["source"]["path"] = str((VIDEOS / target).resolve())
            if (cache["source"]["size"] != (VIDEOS / target).stat().st_size or
                    cache["source"]["mtime_ns"] != (VIDEOS / target).stat().st_mtime_ns):
                raise ValueError(f"Cached video content has changed: {target}")
            write_json(path, cache)
        old_lock, new_lock = CACHE / f"{old_stem}.lock", CACHE / f"{new_stem}.lock"
        if old_lock.exists():
            old_lock.rename(new_lock)
    return changed


def update_benchmark_manifest() -> None:
    manifest = json.loads(BENCHMARK_MANIFEST.read_text())
    source = Path(manifest["source"])
    if not source.is_file():
        raise ValueError(f"Benchmark source data missing: {source}")
    manifest["sha256"] = hashlib.sha256(BENCHMARK_DATA.read_bytes()).hexdigest()
    manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(BENCHMARK_MANIFEST, manifest)


def main() -> None:
    metadata = json.loads(METADATA.read_text())
    mapping = mapping_from_metadata(metadata)
    print(f"Validated {len(mapping)} video names and metadata records", flush=True)
    migrate_videos(metadata, mapping)
    changed_data = migrate_json_references(mapping)
    changed_cache = migrate_cache(mapping)
    update_benchmark_manifest()
    print(f"Migrated {len(mapping)} videos, {changed_data} QA/data files, "
          f"{changed_cache} media caches", flush=True)


if __name__ == "__main__":
    main()
