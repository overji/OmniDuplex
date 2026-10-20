"""Foreground sequential pipeline with validated artifact resume."""
import json
import random
from typing import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from . import generation
from .options import GenerationOptions, BatchOptions, PipelineOptions
from ..domain.catalog import TASK_CATALOG, split_task
from ..domain.common import SAMPLE_SCHEMA
from ..domain.caption import validate_caption
from ..domain.config import PIPELINE_ABILITY_COUNT_WEIGHTS
from ..domain.qa import is_agent_negative
from ..domain.selection import ability_counts, select_tasks
from ..infrastructure.artifacts import digest, file_digest, implementation_digest, write_json
from .merge_qa import load_sample
from .registry import get_task


def run(args: BatchOptions, *, progress: Callable[[str, bool], None] | None = None, videos=None):
    single_video = getattr(args, "video", None)
    is_pipeline = isinstance(args, PipelineOptions)
    if videos is None:
        videos = [single_video] if single_video is not None else sorted(p for p in args.videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"})
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
        manifest = {"layout_version": 3 if is_pipeline else 2, "artifacts": {}}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("invalid batch manifest")
    if is_pipeline and manifest.get('layout_version') != 3:
        raise ValueError('directory pipeline requires the new run layout; create a new run')
    nested = manifest.get("layout_version") in {2, 3}
    if progress is not None:
        progress(f"Run directory: {run_dir}", False)
    settings = {key: getattr(args, key) for key in ("provider", "model", "base_url", "upload_url", "timeout", "retries", "seed", "tasks", "samples_per_task")}
    settings["implementation"] = implementation_digest()
    settings["schema"] = SAMPLE_SCHEMA
    urls = {}
    if is_pipeline:
        from .pipeline import historical_counts, video_urls
        urls = video_urls(args, videos)
        settings["selection"] = {name: getattr(args, name) for name in ("pr_probability", "agent_probability", "agent_positive_probability")}
        settings['ability_count_weights'] = list(PIPELINE_ABILITY_COUNT_WEIGHTS)
        settings["sources"] = [[str(video.relative_to(args.videos_dir)), file_digest(video)] for video in videos]
        settings["videos_dir"] = str(args.videos_dir.resolve())
        settings["num_attempts"] = args.num_attempts
        settings["split"] = args.split
        settings["url_mapping_hash"] = digest(urls)
        if args.resume and manifest.get("pipeline_settings") != settings:
            raise ValueError("pipeline settings or implementation changed; create a new run instead of resuming")
        manifest["pipeline_settings"] = settings
        manifest.setdefault("selections", {})
        manifest.setdefault('attempts', [])
        manifest.setdefault('api_calls', [])
        manifest.setdefault('videos', {})
        for records in (manifest['attempts'], manifest['api_calls']):
            for record in records:
                if record['status'] == 'started':
                    record['status'] = 'interrupted'
                    if records is manifest['api_calls']:
                        usage_path = (run_dir / record['artifact']).with_suffix('.usage.json')
                        try:
                            if usage_path.is_file() and file_digest(usage_path) != record.get('prior_usage_hash'):
                                record['usage'] = json.loads(usage_path.read_text())
                        except (OSError, ValueError):
                            pass
        if "historical_counts" not in manifest:
            manifest["historical_counts"] = historical_counts(run_dir.parent, exclude=run_dir.resolve())
        write_json(manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)
    stats = {task: {"ok": 0, "skip": 0, "failed": 0, "unsuitable": 0, "blocked": 0,
                    "branches": {branch: {"ok": 0, "skip": 0, "failed": 0} for branch in ("positive",)}
                    if task.endswith("_agent") else {}} for task in TASK_CATALOG}
    report = {"schema": SAMPLE_SCHEMA, "tasks": stats, "stages": [], "errors": [], "accepted": 0}
    selected = []
    if is_pipeline and args.resume:
        for key, record in manifest['artifacts'].items():
            path = run_dir / key
            if path.parent.name == 'qa' and path.is_file() and file_digest(path) == record['content_hash']:
                sample = load_sample(path, args.videos_dir)
                if sample is not None:
                    selected.append(path)
    completed_usage = []

    def budget_exhausted():
        return is_pipeline and args.num_attempts is not None and len(manifest['attempts']) >= args.num_attempts

    def stage(video, video_hash, folder, name, index=0, dependency=None, agent_sample_type=None):
        filename = f"{name}.{index:04d}.json" if name in TASK_CATALOG else f"{name}.json"
        stage_dir = folder / ("qa" if name in TASK_CATALOG else ".state") if nested and name != "caption" else folder
        if is_pipeline and name == 'route':
            stage_dir = folder
        output = stage_dir / filename
        key = str(output.relative_to(run_dir))
        fingerprint = digest({"settings": settings, "video": video_hash, "source": str(video.resolve()),
                              "stage": name, "index": index, "dependency": dependency,
                              **({"agent_sample_type": agent_sample_type} if agent_sample_type is not None else {})})
        task_args = GenerationOptions(**{key: getattr(args, key) for key in ("provider", "model", "base_url", "upload_url", "api_key", "timeout", "retries")},
                                       task=name, video=video, input=folder / "caption.json",
                                       usage_output=output.with_suffix(".usage.json"))
        task_args.agent_sample_type = agent_sample_type
        task_args.video_url = urls.get(video.relative_to(args.videos_dir).as_posix())
        state = random.getstate()
        request = None
        branch = None
        usage_record = None
        previous_usage_hash = None
        attempt_record = api_record = None
        reused = False

        def reserve_attempt():
            nonlocal attempt_record
            if is_pipeline and name in TASK_CATALOG and attempt_record is None:
                if budget_exhausted():
                    return False
                attempt_record = {'artifact': key, 'task': name, 'branch': agent_sample_type,
                                  'status': 'started'}
                manifest['attempts'].append(attempt_record)
                write_json(manifest_path, manifest)
            return True
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
                if not reserve_attempt():
                    return None, None
                if is_pipeline:
                    api_record = {'artifact': key, 'task': name, 'status': 'started', 'usage': None,
                                  'prior_usage_hash': previous_usage_hash}
                    manifest['api_calls'].append(api_record)
                    write_json(manifest_path, manifest)
                generated = generation.generate(task_args, request=request)
                result, usage_record = generated.sample, generated.usage_record
                if isinstance(result, dict):
                    result["video"] = str(video.relative_to(args.videos_dir))
                write_json(output, result)
                manifest["artifacts"][key] = {"fingerprint": fingerprint, "content_hash": file_digest(output), "usage": usage_record}
                write_json(manifest_path, manifest)
            completed_usage.append(task_args.usage_output)
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            if attempt_record is not None:
                attempt_record.update(status=status, branch=branch)
            if api_record is not None:
                api_record.update(status=status, usage=usage_record)
            if is_pipeline:
                write_json(manifest_path, manifest)
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
            reserve_attempt()
            if usage_record is None and task_args.usage_output.is_file():
                try:
                    if file_digest(task_args.usage_output) != previous_usage_hash:
                        usage_record = json.loads(task_args.usage_output.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            error = str(exc).replace(args.api_key, '[redacted]') if args.api_key else str(exc)
            if attempt_record is not None:
                attempt_record.update(status='failed', branch=branch, error=error)
            if api_record is not None:
                api_record.update(status='failed', usage=usage_record, error=error)
            if is_pipeline:
                write_json(manifest_path, manifest)
            report["errors"].append({"artifact": key, "error": error, "branch": branch, "usage": usage_record})
            if name in stats:
                stats[name]["failed"] += 1
                if branch:
                    stats[name]["branches"][branch]["failed"] += 1
            if progress is not None:
                progress(f"{key}: failed: {error}", True)
            return None, None
        finally:
            if api_record is not None and api_record['status'] == 'started':
                api_record['status'] = 'interrupted'
                if attempt_record is not None:
                    attempt_record.update(status='interrupted', branch=branch)
                try:
                    if task_args.usage_output.is_file() and file_digest(task_args.usage_output) != previous_usage_hash:
                        api_record['usage'] = json.loads(task_args.usage_output.read_text())
                except (OSError, ValueError):
                    pass
                write_json(manifest_path, manifest)
            random.setstate(state)

    def save_totals():
        bundle = [sample for path in selected
                  if (sample := load_sample(path, args.videos_dir)) is not None]
        report["accepted"] = len(bundle)
        usage_records = [item["usage"] for item in report["stages"] + report["errors"] if item.get("usage")]
        if is_pipeline:
            usage_records = [item['usage'] for item in manifest['api_calls'] if item.get('usage')]
            report['attempted'] = len(manifest['attempts'])
            report['requested_attempts'] = args.num_attempts
            report['stop_reason'] = 'attempt_limit' if budget_exhausted() else 'source_exhausted'
            report['remaining_attempts'] = max(0, args.num_attempts - report['attempted']) if args.num_attempts is not None else None
            report['api_calls'] = manifest['api_calls']
            current_counts = {}
            for sample in bundle:
                current_counts[sample['task']] = current_counts.get(sample['task'], 0) + 1
            report['ability_counts'] = ability_counts(current_counts)
            balancing_counts = dict(manifest['historical_counts'])
            for task, count in current_counts.items():
                balancing_counts[task] = balancing_counts.get(task, 0) + count
            report['balancing_ability_counts'] = ability_counts(balancing_counts)
            report['videos'] = {}
            for relative, video_record in manifest['videos'].items():
                decision = manifest['selections'].get(video_record.get('selection_key'), {})
                attempted = {split_task(a['task'])[0] for a in manifest['attempts']
                             if a['artifact'].split('/')[0] == video_record['folder']}
                successful = {split_task(sample['task'])[0] for sample in bundle if sample['video'] == relative}
                report['videos'][relative] = {
                    'routing_status': video_record.get('routing_status'),
                    'sampled_ability_count': decision.get('sampled_ability_count', 0),
                    'selected_abilities': decision.get('abilities', []),
                    'selected_ability_count': len(decision.get('abilities', [])),
                    'attempted_ability_count': len(attempted),
                    'successful_ability_count': len(successful),
                }
            for label in ('selected', 'attempted', 'successful'):
                report[f'average_{label}_abilities'] = (
                    sum(item[f'{label}_ability_count'] for item in report['videos'].values()) / len(report['videos'])
                    if report['videos'] else 0)
            for task in stats:
                for status in ('ok', 'skip', 'failed', 'interrupted'):
                    stats[task][status] = sum(a['task'] == task and a['status'] == status for a in manifest['attempts'])
                for branch, values in stats[task]['branches'].items():
                    for status in ('ok', 'skip', 'failed'):
                        values[status] = sum(a['task'] == task and a.get('branch') == branch and a['status'] == status
                                            for a in manifest['attempts'])
        priced_records = [item for item in usage_records if item.get("estimated_cost_cny") is not None]
        report["api_usage"] = {
            "calls": len(manifest['api_calls']) if is_pipeline else len(usage_records),
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
        if is_pipeline:
            report['api_usage']['calls_by_stage'] = {stage_name: sum(
                (a['task'] in TASK_CATALOG if stage_name == 'qa' else a['task'] == stage_name)
                for a in manifest['api_calls']) for stage_name in ('caption', 'route', 'qa')}
            report['api_usage']['reused_stages_this_invocation'] = sum(item['reused'] for item in report['stages'])
            report['api_usage']['call_definition'] = 'Logical model generation requests, excluding cache reuse, uploads and internal SDK/HTTP retries.'
            report['api_usage']['missing_usage_calls'] = len(manifest['api_calls']) - report['api_usage']['measured_calls']
            write_json(run_dir / 'api_usage.json', report['api_usage'])
            write_json(run_dir / 'coverage.json', report)

    for video in videos:
        if budget_exhausted():
            break
        relative = str(video.relative_to(args.videos_dir))
        folder = run_dir / f"{video.stem}-{digest(relative)[:12]}"
        if is_pipeline:
            manifest['videos'].setdefault(relative, {'folder': folder.name})
            write_json(manifest_path, manifest)
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
            choices = [{"task": name, "branch": None} for name in args.tasks]
            if is_pipeline:
                selection_key = digest([relative, video_hash, caption, route])
                decision = manifest["selections"].get(selection_key)
                if decision is None:
                    counts = dict(manifest["historical_counts"])
                    for path in selected:
                        existing = json.loads(path.read_text())
                        if is_agent_negative(existing):
                            continue
                        task_name = existing['task']
                        counts[task_name] = counts.get(task_name, 0) + 1
                    decision = select_tasks(route['ranked_tasks'] if suitable else [], counts,
                                            seed=f"{args.seed}:{selection_key}", **settings['selection'])
                    manifest["selections"][selection_key] = decision
                manifest['videos'][relative].update(selection_key=selection_key, routing_status=route['status'])
                write_json(manifest_path, manifest)
                report["selection"] = decision
                report["routing_status"] = route["status"]
                choices = decision["tasks"]
            for choice in choices:
                name = choice["task"]
                if split_task(name)[0] not in suitable:
                    stats[name]["unsuitable"] += args.samples_per_task
                    continue
                for index in range(args.samples_per_task):
                    if budget_exhausted():
                        break
                    result, path = stage(video, video_hash, folder, name, index, dependency=digest([caption, route]), agent_sample_type=choice["branch"])
                    if result is not None and result["status"] == "ok":
                        video_selected.append(path)
                        if is_pipeline and path not in selected:
                            selected.append(path)
            selected.extend(path for path in video_selected if path not in selected)
        finally:
            video_selected = [path for path in selected if path.parent.parent == folder] if is_pipeline else video_selected
            video_bundle = [sample for path in video_selected
                            if (sample := load_sample(path, args.videos_dir)) is not None]
            write_json(folder / "all_qa.json", video_bundle)
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
    save_totals()
    if is_pipeline:
        from .pdf_report import write_pdf_report
        report_path = run_dir / "dataset_report.pdf"
        write_pdf_report(run_dir / "all_qa.json", report_path)
        if progress is not None:
            progress(f"PDF report: {report_path}", False)
    outstanding = {}
    if is_pipeline:
        for record in manifest['api_calls'] + manifest['attempts']:
            outstanding[record['artifact']] = record['status']
    return 1 if report["errors"] or any(status in {'failed', 'interrupted', 'started'}
                                       for status in outstanding.values()) else 0
