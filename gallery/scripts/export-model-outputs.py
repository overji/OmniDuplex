"""Export saved benchmark predictions and official timing scores for the gallery."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.data.files import read_jsonl, write_json
from benchmark.evaluation.timing import match_windows, summarize


def export(config_path, output_dir):
    config = json.loads(config_path.read_text())
    dataset = (config_path.parent / config['dataset']).resolve()
    samples = json.loads(dataset.read_text())
    if not isinstance(samples, list):
        raise ValueError('Benchmark dataset must be an array')
    models = []
    seen_ids = set()
    for run in config['runs']:
        if not run['id'] or run['id'] in seen_ids:
            raise ValueError('Model IDs must be unique and nonempty')
        seen_ids.add(run['id'])
        directory = (config_path.parent / run['path']).resolve()
        predictions = read_jsonl(directory / 'predictions.jsonl')
        if not (directory / 'predictions.jsonl').is_file():
            raise FileNotFoundError(directory / 'predictions.jsonl')
        seen_indices = set()
        records = []
        for prediction in predictions:
            index = prediction['sample_index']
            if type(index) is not int or not 0 <= index < len(samples) or index in seen_indices:
                raise ValueError(f'Invalid or duplicate sample index in {run["id"]}: {index}')
            seen_indices.add(index)
            sample = samples[index]
            if any(prediction.get(key) != sample.get(key) for key in ('video', 'task', 'question')):
                raise ValueError(f'Prediction identity mismatch in {run["id"]}: {index}')
            for event in prediction['events']:
                onset = event.get('onset_seconds')
                if (type(onset) not in (int, float) or not math.isfinite(onset)
                        or onset < 0 or not isinstance(event.get('text'), str)):
                    raise ValueError(f'Invalid prediction event in {run["id"]}: {index}')
            records.append({'prediction': prediction, 'score': match_windows(sample, prediction['events'])})
        metrics = summarize(samples, predictions)
        models.append({'id': run['id'], 'name': run['name'], 'evaluatedSamples': len(predictions),
                       'totalSamples': len(samples), 'overall': metrics['groups'].get('overall'),
                       'records': records})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'results.json', {'version': 1,
        'datasetSha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'samples': samples, 'models': models})
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'model-runs.json')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'model_outputs')
    args = parser.parse_args()
    for model in export(args.config.resolve(), args.output_dir):
        print(f'{model["name"]}: {model["evaluatedSamples"]}/{model["totalSamples"]} samples')
