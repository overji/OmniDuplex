"""Foreground, sequential generation with validated content-addressed resume."""
import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from . import cli
from .catalog import TASK_CATALOG
from .common import SAMPLE_SCHEMA
from .merge_qa import load_sample
from .registry import get_task
from .video_caption import validate_caption

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_digest():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "prompts").rglob("*.md"))
    return digest([(str(p.relative_to(ROOT)), file_digest(p)) for p in paths])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_CATALOG), default=list(TASK_CATALOG))
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    # Keep API defaults identical to the single-task CLI.
    defaults = cli.parse_args(["--task", "caption", "--video", "unused.mp4"])
    for name in ("model", "base_url", "api_key", "timeout", "retries"):
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(defaults, name),
                            type=float if name == "timeout" else int if name == "retries" else str)
    args = parser.parse_args(argv)
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be positive")
    args.tasks = list(dict.fromkeys(args.tasks))
    return args


def run(args):
    videos = sorted(p for p in args.videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"})
    if not videos:
        raise ValueError("no source videos found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if args.resume and manifest_path.exists() else {"artifacts": {}}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("invalid batch manifest")
    settings = {key: getattr(args, key) for key in ("model", "base_url", "timeout", "retries", "seed", "tasks", "samples_per_task")}
    settings["implementation"] = implementation_digest()
    settings["schema"] = SAMPLE_SCHEMA
    stats = {task: {"ok": 0, "skip": 0, "failed": 0, "unsuitable": 0, "blocked": 0,
                    "branches": {branch: {"ok": 0, "skip": 0, "failed": 0} for branch in ("positive", "no_context", "missing_tool")}
                    if task.endswith("_agent") else {}} for task in TASK_CATALOG}
    report = {"schema": SAMPLE_SCHEMA, "tasks": stats, "stages": [], "errors": [], "accepted": 0}
    selected = []

    def stage(video, video_hash, folder, name, index=0, dependency=None):
        filename = f"{name}.{index:04d}.json" if name in TASK_CATALOG else f"{name}.json"
        output = folder / filename
        key = str(output.relative_to(args.output_dir))
        fingerprint = digest({"settings": settings, "video": video_hash, "source": str(video.resolve()),
                              "stage": name, "index": index, "dependency": dependency})
        task_args = argparse.Namespace(**{key: getattr(args, key) for key in ("model", "base_url", "api_key", "timeout", "retries")},
                                       task=name, video=video, input=folder / "caption.json", print_prompt=False)
        state = random.getstate()
        request = None
        branch = None
        try:
            random.seed(int(fingerprint, 16))
            request = cli.prepare_request(task_args)
            if name.endswith("_agent"):
                branch = request.context.sample_type
            old = manifest["artifacts"].get(key, {})
            reused = False
            if args.resume and old.get("fingerprint") == fingerprint and output.is_file():
                try:
                    if old.get("content_hash") != file_digest(output):
                        raise ValueError("artifact content changed")
                    result = json.loads(output.read_text())
                    if name == "caption":
                        validate_caption(result, request.duration_seconds)
                    else:
                        if result.get("schema") != SAMPLE_SCHEMA or result.get("task") != name:
                            raise ValueError("cached schema/task mismatch")
                        expected_context = json.loads(json.dumps(asdict(request.context)))
                        if result.get("generation_context") != expected_context:
                            raise ValueError("cached generation context mismatch")
                        get_task(name).validate_result(result, request.context)
                    reused = True
                except (ValueError, TypeError, KeyError):
                    reused = False
            if not reused:
                result = cli.generate(task_args, request=request)
                if isinstance(result, dict):
                    result["video"] = str(video.relative_to(args.videos_dir))
                write_json(output, result)
                manifest["artifacts"][key] = {"fingerprint": fingerprint, "content_hash": file_digest(output)}
                write_json(manifest_path, manifest)
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            report["stages"].append({"artifact": key, "status": status, "reused": reused, "branch": branch})
            if name in stats:
                stats[name][status] += 1
                if branch:
                    stats[name]["branches"][branch][status] += 1
            print(f"{key}: {status}" + (" (reused)" if reused else ""), flush=True)
            return result, output
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            report["errors"].append({"artifact": key, "error": str(exc), "branch": branch})
            if name in stats:
                stats[name]["failed"] += 1
                if branch:
                    stats[name]["branches"][branch]["failed"] += 1
            print(f"{key}: failed: {exc}", file=sys.stderr, flush=True)
            return None, None
        finally:
            random.setstate(state)

    for video in videos:
        relative = str(video.relative_to(args.videos_dir))
        folder = args.output_dir / f"{video.stem}-{digest(relative)[:12]}"
        try:
            video_hash = file_digest(video)
        except OSError as exc:
            report["errors"].append({"video": relative, "error": str(exc)})
            for name in args.tasks:
                stats[name]["blocked"] += args.samples_per_task
            continue
        caption, _ = stage(video, video_hash, folder, "caption")
        if caption is None:
            for name in args.tasks:
                stats[name]["blocked"] += args.samples_per_task
            continue
        route, _ = stage(video, video_hash, folder, "route", dependency=digest(caption))
        if route is None:
            for name in args.tasks:
                stats[name]["blocked"] += args.samples_per_task
            continue
        suitable = {item["task"] for item in route["ranked_tasks"]} if route["status"] == "ok" else set()
        for name in args.tasks:
            if name not in suitable:
                stats[name]["unsuitable"] += args.samples_per_task
                continue
            for index in range(args.samples_per_task):
                result, path = stage(video, video_hash, folder, name, index, dependency=digest([caption, route]))
                if result is not None and result["status"] == "ok":
                    selected.append(path)
        write_json(args.output_dir / "coverage.json", report)
    # Merge only accepted artifacts from THIS invocation; never stale prior files.
    bundle = [load_sample(path, args.videos_dir) for path in selected]
    report["accepted"] = len(bundle)
    write_json(args.output_dir / "all_qa.json", bundle)
    write_json(args.output_dir / "coverage.json", report)
    return 1 if report["errors"] else 0


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
