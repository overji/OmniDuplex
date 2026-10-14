"""Timing-only one-to-one matching of response onsets to inclusive windows."""
from collections import defaultdict

from benchmark.data.timestamps import seconds
from benchmark.data.files import write_json


def is_scored_response(sample, response):
    if sample.get("mode") != "agent":
        return True
    if response.get("role") == "tool":
        return False
    kind = sample.get("sample_type", sample.get("generation_context", {}).get("sample_type"))
    if kind in {"missing_tool", "no_context"}:
        return True
    return response.get("role") == "assistant" and any(
        isinstance(call, dict) and (call.get("function") or {}).get("name") == "agent_call"
        for call in (response.get("tool_calls") or []))


def reference_windows(sample):
    return [(index, response) for index, response in enumerate(sample["responses"])
            if is_scored_response(sample, response)
            and "window_start" in response and "window_end" in response]


def match_windows(sample, events):
    windows = [(index, seconds(response["window_start"]), seconds(response["window_end"]))
               for index, response in reference_windows(sample)]
    remaining = set(range(len(events)))
    matches = []
    # Earliest-deadline matching maximizes interval/point matches without reusing an onset.
    for response_index, start, end in sorted(windows, key=lambda row: (row[2], row[1])):
        candidates = [i for i in remaining if start <= events[i]["onset_seconds"] <= end]
        if candidates:
            event_index = min(candidates, key=lambda i: events[i]["onset_seconds"])
            remaining.remove(event_index)
            matches.append({"response_index": response_index, "event_index": event_index,
                            "onset_seconds": events[event_index]["onset_seconds"],
                            "window_start_seconds": start, "window_end_seconds": end})
    return {"windows": len(windows), "predictions": len(events), "hits": len(matches),
            "misses": len(windows) - len(matches), "extra_predictions": len(remaining),
            "matches": sorted(matches, key=lambda row: row["response_index"])}


def summarize(samples, predictions):
    groups = defaultdict(lambda: {"samples": 0, "windows": 0, "predictions": 0, "hits": 0})
    per_sample = []
    for prediction in predictions:
        sample = samples[prediction["sample_index"]]
        score = match_windows(sample, prediction["events"])
        per_sample.append({"sample_index": prediction["sample_index"], "task": sample["task"], **score})
        group_names = ["overall", f"mode:{sample['mode']}", f"ability:{sample['ability']}"]
        if sample.get("sample_type"):
            group_names.append(f"agent:{sample['sample_type']}")
        for group in group_names:
            for key in ["windows", "predictions", "hits"]:
                groups[group][key] += score[key]
            groups[group]["samples"] += 1
    for score in groups.values():
        score["window_hit_rate"] = score["hits"] / score["windows"] if score["windows"] else None
        score["timing_precision"] = score["hits"] / score["predictions"] if score["predictions"] else 0.0
        recall = score["window_hit_rate"] or 0.0
        precision = score["timing_precision"]
        score["timing_f1"] = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"evaluation": "timing_only_inclusive_windows_one_to_one_onsets",
            "total_benchmark_samples": len(samples), "evaluated_samples": len(predictions),
            "excluded_reference_responses_without_windows": sum(
                "window_start" not in r or "window_end" not in r for s in samples for r in s["responses"]),
            "excluded_agent_non_trigger_responses": sum(
                not is_scored_response(s, r) for s in samples for r in s["responses"]),
            "groups": dict(groups), "per_sample": per_sample}


def write_metrics(samples, predictions, path):
    result = summarize(samples, predictions)
    write_json(path, result)
    return result
