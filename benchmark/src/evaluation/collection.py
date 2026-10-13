"""Combine completed realtime shards and recompute timing metrics."""
import json

from benchmark.evaluation.timing import reference_windows, write_metrics
from benchmark.data.files import atomic_write_text, read_jsonl


def execute(args):
    samples = json.loads(args.data.read_text())
    metrics = collect(samples, args.runs, args.output_dir)
    print(json.dumps(metrics['groups'].get('overall', {}), ensure_ascii=False))


def collect(samples, runs, output_dir, *, live=False, expected_indices=None):
    predictions = {}
    for run in runs:
        path = run / 'predictions.jsonl'
        if not path.exists():
            continue
        for prediction in read_jsonl(path, complete_only=live):
            index = prediction['sample_index']
            if index in predictions:
                raise ValueError(f'Duplicate sample {index} in input runs')
            predictions[index] = prediction
    ordered = [predictions[index] for index in sorted(predictions)]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / 'predictions.jsonl', ''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in ordered))
    metrics = write_metrics(samples, ordered, output_dir / 'metrics.json')
    expected = set(range(len(samples))) if expected_indices is None else set(expected_indices)
    atomic_write_text(output_dir / 'collection_config.json', json.dumps({
        'runs': [str(run.resolve()) for run in runs],
        'completed_sample_indices': sorted(predictions),
        'missing_sample_indices': sorted(expected - predictions.keys()),
    }, indent=2) + '\n')
    report = [f"# {output_dir.name} benchmark", "",
              f"Evaluated samples: {len(ordered)} / {len(samples)}", "",
              "Timing only; one-to-one response onsets inside inclusive reference windows.", "",
              "| Group | Samples | Windows | Predictions | Hits | Hit rate | Precision | F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, score in metrics['groups'].items():
        rate = score['window_hit_rate']
        rate_text = f"{rate:.2%}" if rate is not None else "N/A"
        report.append(f"| {name} | {score['samples']} | {score['windows']} | {score['predictions']} | "
                      f"{score['hits']} | {rate_text} | {score['timing_precision']:.2%} | {score['timing_f1']:.2%} |")
    scores = {row['sample_index']: row for row in metrics['per_sample']}
    for prediction in ordered:
        index = prediction['sample_index']
        sample = samples[index]
        score = scores[index]
        matched = {row['event_index']: row['response_index'] for row in score['matches']}
        report.extend(["", f"## Sample {index}: {sample['task']}", "", sample['question'], "",
                       f"Video: {sample['video']}; question time: {sample['question_time']}; "
                       f"hits: {score['hits']}/{score['windows']}.", "",
                       "Reference windows: " + "; ".join(
                           f"{r['window_start']}–{r['window_end']}" for _, r in reference_windows(sample)), "",
                       "| Onset (s) | Window match | Generated text |", "| ---: | --- | --- |"])
        for event_index, event in enumerate(prediction['events']):
            text = event['text'].strip().replace("|", "\\|").replace("\n", " ")
            match = f"reference {matched[event_index]}" if event_index in matched else "miss"
            report.append(f"| {event['onset_seconds']:.3f} | {match} | {text} |")
        if not prediction['events']:
            report.append("| — | — | No response generated |")
    atomic_write_text(output_dir / 'report.md', "\n".join(report) + "\n")
    return metrics
