"""Create a side-by-side timing metric table for completed model runs."""
import csv
import json


def execute(args):
    rows = []
    for run in args.runs:
        metrics_path = run / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text())
        score = metrics["groups"]["overall"]
        rows.append({"model": run.name,
            "samples": f"{metrics['evaluated_samples']}/{metrics['total_benchmark_samples']}",
            "windows": score["windows"], "predictions": score["predictions"], "hits": score["hits"],
            "hit_rate": score["window_hit_rate"], "precision": score["timing_precision"],
            "f1": score["timing_f1"]})
    judge_dir = getattr(args, "judge_dir", None)
    if judge_dir:
        for row in rows:
            path = judge_dir / row["model"] / "judge_metrics.json"
            judged = json.loads(path.read_text()) if path.exists() else {}
            score = judged.get("groups", {}).get("overall", {})
            row.update(judge_status=judged.get("status", "not_run"),
                       response_mean_matched=score.get("response_mean_matched"),
                       response_mean_all_windows=score.get("response_mean_all_windows"),
                       call_mean_all_windows=score.get("call_mean_all_windows"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Realtime timing benchmark comparison", "",
        "Scores use one-to-one onset matching against inclusive reference windows.", "",
        "| Model | Samples | Windows | Predictions | Hits | Hit rate | Precision | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        rate = lambda x: "N/A" if x is None else f"{x:.2%}"
        lines.append(f"| {row['model']} | {row['samples']} | {row['windows']} | {row['predictions']} | "
            f"{row['hits']} | {rate(row['hit_rate'])} | {rate(row['precision'])} | {rate(row['f1'])} |")
    if judge_dir:
        lines.extend(["", "Content scores (0–2; N/A includes incomplete judge runs):", "",
            "| Model | Judge status | Matched response | All-window response | Agent call |",
            "| --- | --- | ---: | ---: | ---: |"])
        for row in rows:
            fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
            lines.append(f"| {row['model']} | {row['judge_status']} | " + " | ".join(
                fmt(row[k]) for k in ("response_mean_matched", "response_mean_all_windows",
                                      "call_mean_all_windows")) + " |")
    lines.extend(["", "Subgroup metrics remain in each model's `metrics.json`."])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output.with_suffix(".md"))
