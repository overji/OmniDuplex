"""Foreground sequential pipeline with validated artifact resume."""
import json
import random
from typing import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from . import generation
from .options import GenerationOptions, BatchOptions
from ..domain.catalog import TASK_CATALOG, split_task
from ..domain.common import SAMPLE_SCHEMA
from ..domain.caption import validate_caption
from ..infrastructure.artifacts import digest, file_digest, implementation_digest, write_json
from .merge_qa import load_sample
from .registry import get_task


def run(args: BatchOptions, *, progress: Callable[[str, bool], None] | None = None):
    videos = sorted(p for p in args.videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"})
    if not videos:
        raise ValueError("no source videos found")
    if args.resume:
        run_dir = args.output_dir
        manifest_path = run_dir / ".state" / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("resume requires --output-dir pointing to a run with a manifest")
        manifest = json.loads(manifest_path.read_text())
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                continue
        manifest_path = run_dir / ".state" / "manifest.json"
        manifest = {"layout_version": 2, "artifacts": {}}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("invalid batch manifest")
    nested = manifest.get("layout_version") == 2
    write_json(manifest_path, manifest)
    if progress is not None:
        progress(f"Run directory: {run_dir}", False)
    settings = {key: getattr(args, key) for key in ("provider", "model", "base_url", "upload_url", "timeout", "retries", "seed", "tasks", "samples_per_task")}
    settings["implementation"] = implementation_digest()
    settings["schema"] = SAMPLE_SCHEMA
    stats = {task: {"ok": 0, "skip": 0, "failed": 0, "unsuitable": 0, "blocked": 0,
                    "branches": {branch: {"ok": 0, "skip": 0, "failed": 0} for branch in ("positive", "no_context", "missing_tool")}
                    if task.endswith("_agent") else {}} for task in TASK_CATALOG}
    report = {"schema": SAMPLE_SCHEMA, "tasks": stats, "stages": [], "errors": [], "accepted": 0}
    selected = []
    completed_usage = []

    def stage(video, video_hash, folder, name, index=0, dependency=None):
        filename = f"{name}.{index:04d}.json" if name in TASK_CATALOG else f"{name}.json"
        stage_dir = folder / ("qa" if name in TASK_CATALOG else ".state") if nested and name != "caption" else folder
        output = stage_dir / filename
        key = str(output.relative_to(run_dir))
        fingerprint = digest({"settings": settings, "video": video_hash, "source": str(video.resolve()),
                              "stage": name, "index": index, "dependency": dependency})
        task_args = GenerationOptions(**{key: getattr(args, key) for key in ("provider", "model", "base_url", "upload_url", "api_key", "timeout", "retries")},
                                       task=name, video=video, input=folder / "caption.json",
                                       usage_output=output.with_suffix(".usage.json"))
        state = random.getstate()
        request = None
        branch = None
        usage_record = None
        previous_usage_hash = None
        try:
            previous_usage_hash = file_digest(task_args.usage_output) if task_args.usage_output.is_file() else None
            random.seed(int(fingerprint, 16))
            request = generation.prepare_request(task_args)
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
                    usage_path = task_args.usage_output
                    if usage_path.is_file():
                        usage_record = json.loads(usage_path.read_text(encoding="utf-8"))
                except (ValueError, TypeError, KeyError):
                    reused = False
            if reused:
                usage_record = old.get("usage", usage_record)
                old["usage"] = usage_record
                write_json(manifest_path, manifest)
            if not reused:
                generated = generation.generate(task_args, request=request)
                result, usage_record = generated.sample, generated.usage_record
                if isinstance(result, dict):
                    result["video"] = str(video.relative_to(args.videos_dir))
                write_json(output, result)
                manifest["artifacts"][key] = {"fingerprint": fingerprint, "content_hash": file_digest(output), "usage": usage_record}
                write_json(manifest_path, manifest)
            completed_usage.append(task_args.usage_output)
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            report["stages"].append({"artifact": key, "status": status, "reused": reused,
                                      "branch": branch, "usage": usage_record})
            if name in stats:
                stats[name][status] += 1
                if branch:
                    stats[name]["branches"][branch][status] += 1
            if progress is not None:
                progress(f"{key}: {status}" + (" (reused)" if reused else ""), False)
            return result, output
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            if usage_record is None and task_args.usage_output.is_file():
                try:
                    if file_digest(task_args.usage_output) != previous_usage_hash:
                        usage_record = json.loads(task_args.usage_output.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            report["errors"].append({"artifact": key, "error": str(exc), "branch": branch, "usage": usage_record})
            if name in stats:
                stats[name]["failed"] += 1
                if branch:
                    stats[name]["branches"][branch]["failed"] += 1
            if progress is not None:
                progress(f"{key}: failed: {exc}", True)
            return None, None
        finally:
            random.setstate(state)

    def save_totals():
        bundle = [load_sample(path, args.videos_dir) for path in selected]
        report["accepted"] = len(bundle)
        usage_records = [item["usage"] for item in report["stages"] + report["errors"] if item.get("usage")]
        priced_records = [item for item in usage_records if item.get("estimated_cost_cny") is not None]
        report["api_usage"] = {
            "calls": len(usage_records),
            "measured_calls": sum(item.get("status") == "measured" for item in usage_records),
            "unavailable_calls": sum(item.get("status") == "unavailable" for item in usage_records),
            "input_tokens": sum(item.get("input_tokens") or 0 for item in usage_records),
            "output_tokens": sum(item.get("output_tokens") or 0 for item in usage_records),
            "total_tokens": sum(item.get("total_tokens") or 0 for item in usage_records),
            "estimated_cost_cny": round(sum(item["estimated_cost_cny"] for item in priced_records), 10) if priced_records else None,
            "cost_note": "Only provider-reported usage with a configured price is included; unavailable costs remain null.",
        }
        write_json(run_dir / "all_qa.json", bundle)
        write_json(run_dir / "coverage.json", report)

    for video in videos:
        relative = str(video.relative_to(args.videos_dir))
        folder = run_dir / f"{video.stem}-{digest(relative)[:12]}"
        video_selected = []
        completed_usage.clear()
        try:
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
                if split_task(name)[0] not in suitable:
                    stats[name]["unsuitable"] += args.samples_per_task
                    continue
                for index in range(args.samples_per_task):
                    result, path = stage(video, video_hash, folder, name, index, dependency=digest([caption, route]))
                    if result is not None and result["status"] == "ok":
                        video_selected.append(path)
            selected.extend(video_selected)
        finally:
            write_json(folder / "all_qa.json", [load_sample(path, args.videos_dir) for path in video_selected])
            save_totals()
            # Only sidecars for validated, persisted stages are disposable.
            for usage_path in completed_usage:
                try:
                    usage_path.unlink(missing_ok=True)
                except OSError as exc:
                    report["errors"].append({"artifact": str(usage_path.relative_to(run_dir)), "error": f"cleanup failed: {exc}"})
                    if progress is not None:
                        progress(f"{usage_path}: cleanup failed: {exc}", True)
            write_json(run_dir / "coverage.json", report)
    return 1 if report["errors"] else 0
