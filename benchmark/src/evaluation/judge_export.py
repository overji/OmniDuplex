"""Freeze explicitly selected model results into one reference-only judge package."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchmark.config.paths import ROOT as BENCHMARK_ROOT
from benchmark.data.agent_schema import adapt_sample
from benchmark.data.files import write_json
from benchmark.evaluation import judge, probe_judge
from benchmark.evaluation.judge_package import SCHEMA, content_hash, task_id, needs_api, validate, inspect

ROOT = BENCHMARK_ROOT.parent


def load_predictions(path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.strip() and line.endswith(b'\n')]
    return records, {'file': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                     'sha256': hashlib.sha256(raw).hexdigest(), 'records': len(records),
                     'incomplete_tail_ignored': bool(raw and not raw.endswith(b'\n'))}


def export(sources, output):
    config = json.loads(sources.read_text())
    def path(value):
        p = Path(value)
        return p if p.is_absolute() else ROOT / p
    data = path(config['data']); raw = data.read_bytes()
    data_hash = hashlib.sha256(raw).hexdigest()
    if data_hash != config['data_sha256']:
        raise ValueError('Canonical dataset identity changed')
    samples = json.loads(raw)
    manifest_path = path(config['manifest']); manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != config['manifest_sha256']:
        raise ValueError('Canonical Probe manifest changed')
    manifest = [json.loads(line) for line in manifest_raw.splitlines() if line.strip()]
    probe_judge.validate_manifest(samples, manifest, data_hash)
    adapted = [adapt_sample(s, 'task-start') for s in samples]
    system = path(config['system_prompt']).read_text()
    bundle = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset': {'sha256': data_hash, 'canonical_qa': len(samples),
                          'public_qa': config['public_qa'], 'manifest_sha256': config['manifest_sha256']},
              'prompts': {name: {'text': text, 'sha256': judge.fingerprint(text)}
                          for name, text in [('realtime', system), ('probe', system + '\n' + probe_judge.PROBE_RUBRIC)]},
              'runs': [], 'tasks': []}
    for entry in config['runs']:
        run = {k: entry[k] for k in ('run_id', 'model', 'mode', 'variant')}
        if 'input_modality' in entry:run['input_modality'] = entry['input_modality']
        run['sources'] = []
        if run['mode'] == 'realtime':
            merged = {}
            for source in entry['sources']:
                records, identity = load_predictions(path(source['file']))
                run['sources'].append({**identity, 'selection': source['selection']})
                seen = set()
                for prediction in records:
                    if prediction.get('status') != 'complete':
                        continue
                    index = prediction['sample_index']
                    if type(index) is not int or not 0 <= index < len(samples) or index in seen:
                        raise ValueError('Invalid or duplicate realtime source index')
                    seen.add(index)
                    selector = source['selection']
                    if selector not in ('all', 'agent', 'non-agent'):
                        raise ValueError('Unknown realtime source selection')
                    if selector == 'agent' and samples[index]['mode'] != 'agent':continue
                    if selector == 'non-agent' and samples[index]['mode'] == 'agent':continue
                    # Ordered source precedence is explicit in the committed registry.
                    merged[index] = prediction
            predictions = [merged[i] for i in sorted(merged)]
            rows, _ = judge.prepare_rows(adapted, predictions)
            run['coverage'] = dict(completed_qa=len(predictions), expected_public_qa=config['public_qa'],
                                   reference_rows=len(rows))
        elif run['mode'] == 'probe':
            records, identity = load_predictions(path(entry['predictions']))
            run['sources'].append(identity)
            predictions = [r for r in records if r.get('status') == 'complete']
            rows, _, counts = probe_judge.prepare_rows(samples, manifest, predictions)
            run['coverage'] = dict(completed_requests=len(predictions), manifest_requests=len(manifest),
                                   decisions=counts, tp_rows=len(rows))
        else:
            raise ValueError('Unknown evaluation mode')
        if not rows:
            raise ValueError('Selected run has no judge rows: ' + run['run_id'])
        for row in sorted(rows, key=lambda r: (r['sample_index'], r['response_index'])):
            sample = samples[row['sample_index']]
            task = dict(run_id=run['run_id'], rubric=run['mode'], qa_sha256=judge.fingerprint(sample),
                        video=sample['video'], row=row, api_required=needs_api(row),
                        matched=row.get('event_index', 0) is not None)
            task['task_id'] = task_id(task)
            bundle['tasks'].append(task)
        run['tasks'] = len(rows)
        bundle['runs'].append(run)
    bundle['bundle_id'] = content_hash(bundle)
    validate(bundle)
    if output.exists():
        previous = validate(json.loads(output.read_text()))
        if previous['bundle_id'] != bundle['bundle_id']:
            raise ValueError('Output belongs to a different source snapshot; choose a new file')
        bundle = previous
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, bundle)
    return bundle


def execute(args):
    bundle = export(args.sources, args.output)
    print(json.dumps(inspect(bundle), indent=2))
    print(json.dumps(Counter(r['mode'] for r in bundle['runs'])))
