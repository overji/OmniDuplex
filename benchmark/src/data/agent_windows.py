"""Explicit, backed-up migration of legacy Agent reference windows."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math

from benchmark.data.files import atomic_write_text, write_json
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows


POLICY = "agent-reference-windows-v1"


def timestamp(value):
    return f"{value // 60:02d}:{value % 60:02d}"


def migrate(samples):
    updated = copy.deepcopy(samples)
    changes = []
    for sample_index, sample in enumerate(updated):
        if sample.get("mode") != "agent":
            continue
        limit = math.floor(sample["generation_context"]["duration_seconds"])
        kind = sample.get("sample_type", sample["generation_context"].get("sample_type", "positive"))
        latest_tool = None
        for response_index, response in enumerate(sample["responses"]):
            before = copy.deepcopy(response)
            at = int(seconds(response["trigger_time"]))
            if response.get("role") == "tool":
                latest_tool = response["trigger_time"]
                continue
            if kind in {"missing_tool", "no_context"}:
                if at != seconds(sample["question_time"]):
                    raise ValueError("Refusal timestamp must equal question_time")
                response.update(window_start=timestamp(at), window_end=timestamp(at + 3))
            elif response.get("tool_calls"):
                latest_tool = None
                if response.get("window_start") == response.get("window_end"):
                    response["window_end"] = timestamp(at + 3)
            elif response.get("role") == "assistant" and (response.get("agent_result_time") or latest_tool):
                result_time = response.get("agent_result_time", latest_tool)
                result_at = int(seconds(result_time))
                response["agent_result_time"] = result_time
                if "window_start" not in response and "window_end" not in response:
                    response.update(trigger_time=timestamp(result_at + 1),
                                    window_start=timestamp(result_at + 1),
                                    window_end=timestamp(result_at + 3))
                if "window_start" not in response or "window_end" not in response:
                    raise ValueError("Completion window must contain both boundaries")
                if seconds(response["window_start"]) <= result_at:
                    raise ValueError("Completion window must follow its own tool result")
            else:
                raise ValueError("Cannot associate Agent assistant response with a tool result")
            start = seconds(response["window_start"])
            end = seconds(response["window_end"])
            if not 0 <= start <= seconds(response["trigger_time"]) <= end <= limit:
                raise ValueError(f"Invalid in-video window for sample {sample_index}, response {response_index}")
            if response != before:
                changes.append({"sample_index": sample_index, "original_response_index": response_index,
                                "before": before, "after": copy.deepcopy(response)})
        # Association was established above, before moving completion timestamps.
        sample["responses"].sort(key=lambda response: seconds(response["trigger_time"]))
    return updated, changes


def execute(args):
    original = args.data.read_bytes()
    samples = json.loads(original)
    updated, changes = migrate(samples)
    counts = {"samples": len(updated), "responses": sum(len(s["responses"]) for s in updated),
              "reference_windows": sum("window_start" in r and "window_end" in r
                                       for s in updated for r in s["responses"]),
              "scored_windows": sum(len(reference_windows(s)) for s in updated),
              "changed_responses": len(changes)}
    if args.dry_run or not changes:
        print(json.dumps({**counts, "dry_run": args.dry_run, "policy": POLICY}))
        return
    original_manifest = args.manifest.read_bytes()
    old_hash = hashlib.sha256(original).hexdigest()
    manifest = json.loads(original_manifest)
    if manifest["sha256"] != old_hash:
        raise ValueError("Dataset does not match manifest; resolve before migrating")
    now = datetime.now(timezone.utc)
    backup_dir = args.backup_dir or args.data.parent / "results" / ("agent_windows_" + now.strftime("%Y%m%dT%H%M%S%fZ"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / "all_qa.original.json"
    backup.write_bytes(original)
    (backup_dir / "data_manifest.original.json").write_bytes(original_manifest)
    ledger = backup_dir / "migration.json"
    new_text = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
    new_hash = hashlib.sha256(new_text.encode()).hexdigest()
    write_json(ledger, {"policy": POLICY, "created_at": now.isoformat(), "data": str(args.data.resolve()),
                        "original_sha256": old_hash, "migrated_sha256": new_hash,
                        "counts": counts, "changes": changes, "rescored": False})
    manifest.update(source_sha256=manifest.get("source_sha256", old_hash), sha256=new_hash,
                    samples=len(updated), window_policy=POLICY,
                    migration={"original_sha256": old_hash, "backup": str(backup.resolve()),
                               "ledger": str(ledger.resolve())})
    # Both originals are backed up before replacing either file.
    try:
        atomic_write_text(args.data, new_text)
        write_json(args.manifest, manifest)
    except Exception:
        atomic_write_text(args.data, original.decode())
        atomic_write_text(args.manifest, original_manifest.decode())
        raise
    print(json.dumps({**counts, "policy": POLICY, "backup": str(backup), "ledger": str(ledger)}))
