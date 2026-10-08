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
    lines.extend(["", "Subgroup metrics remain in each model's `metrics.json`."])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(args.output.with_suffix(".md"))
