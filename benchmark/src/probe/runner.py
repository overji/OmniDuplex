"""Prepare, run and score independent prefix probes with incremental resume."""
import fcntl
import json
import math
from pathlib import Path
import time

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.probe.data import SCHEMA, digest, file_hash, load_manifest, prepare
from benchmark.probe.media import prefix_clip
from benchmark.probe.metrics import parse_decision, summarize, write_report
from benchmark.probe.prompts import model_input


def selected_rows(rows, max_pairs):
    if max_pairs < 0:
        raise ValueError('max-pairs must be nonnegative')
    pairs = list(dict.fromkeys(row['pair_id'] for row in rows))
    chosen = set(pairs[:max_pairs] if max_pairs else pairs)
    return [row for row in rows if row['pair_id'] in chosen]


def model_identity(model_path):
    if not model_path.is_dir() or not (model_path / 'config.json').is_file():
        raise ValueError('model-path must be a local checkpoint directory; probe never downloads weights')
    files = {name: file_hash(model_path / name) for name in
             ('config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
              'preprocessor_config.json', 'video_preprocessor_config.json', 'processor_config.json',
              'chat_template.json', 'chat_template.jinja',
              'model.safetensors.index.json', 'pytorch_model.bin.index.json')
             if (model_path / name).is_file()}
    weights = [[path.name, path.stat().st_size, path.stat().st_mtime_ns]
               for path in sorted([*model_path.glob('*.safetensors'), *model_path.glob('pytorch_model*.bin')])]
    return {'path': str(model_path.resolve()), 'config_hashes': files, 'weight_file_stats': weights}


def run_config(args):
    if (not math.isfinite(args.fps) or args.fps <= 0 or args.max_pixels < 1
            or args.max_new_tokens < 1 or args.max_pairs < 0):
        raise ValueError('Invalid probe sampling or generation setting')
    implementation = {path.name: file_hash(path) for path in Path(__file__).parent.glob('*.py')}
    return {'schema': SCHEMA, 'manifest_sha256': file_hash(args.manifest),
        'prompt_sha256': file_hash(args.system_prompt), 'implementation': digest(implementation),
        'model': args.probe_model, 'model_identity': model_identity(args.model_path),
        'fps': args.fps, 'max_pixels': args.max_pixels, 'max_new_tokens': args.max_new_tokens,
        'dtype': args.dtype, 'device_map': args.device_map, 'attn_implementation': args.attn_implementation,
        'max_pairs': args.max_pairs, 'do_sample': False, 'return_audio': False,
        'independent_requests': True, 'audio': 'use_source_track_if_present'}


def run(args):
    rows = selected_rows(load_manifest(args.manifest), args.max_pairs)
    config = run_config(args)
    template = args.system_prompt.read_text()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.probe.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process owns this probe output directory') from exc
        return run_locked(args, rows, config, template)


def run_locked(args, rows, config, template):
    config_path = args.output_dir / 'probe_config.json'
    prediction_path = args.output_dir / 'probe_predictions.jsonl'
    error_path = args.output_dir / 'probe_errors.jsonl'
    prompt_path = args.output_dir / 'probe_prompts.jsonl'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('Probe configuration changed; use a new output directory')
    elif any(path.exists() for path in (prediction_path, error_path, prompt_path)):
        raise ValueError('Existing probe artifacts have no run config; use a new output directory')
    write_json(config_path, config)
    for path in (prediction_path, error_path, prompt_path):
        recover_jsonl_tail(path)
    records, errors = read_jsonl(prediction_path), read_jsonl(error_path)
    summarize(rows, records, errors)  # Reject mixed or duplicate records before loading weights.
    completed = {record['probe_id'] for record in records}
    pending = [row for row in rows if row['probe_id'] not in completed]
    if not pending:
        write_report(rows, records, args.output_dir, errors)
        return 0
    from benchmark.probe import models
    state = models.load(args)
    for row in pending:
        started = time.perf_counter()
        try:
            clip = prefix_clip(row, args.cache_dir)
            probe = model_input(row, clip, template)
            append_jsonl(prompt_path, {'probe_id': row['probe_id'], 'video_prefix': str(clip),
                'system_prompt': probe.system_prompt, 'user_prompt': probe.user_prompt})
            raw = models.generate(args, state, probe)
            if not isinstance(raw, str):
                raise TypeError('Probe adapter did not return text')
        except Exception as exc:
            record = {'probe_id': row['probe_id'], 'pair_id': row['pair_id'],
                'error_type': type(exc).__name__, 'error': str(exc),
                'wall_seconds': time.perf_counter() - started}
            append_jsonl(error_path, record)
            errors.append(record)
            print(f"PROBE ERROR {row['probe_id']}: {type(exc).__name__}: {exc}", flush=True)
        else:
            decision = parse_decision(raw)
            record = {'probe_id': row['probe_id'], 'pair_id': row['pair_id'],
                'sample_index': row['sample_index'], 'response_index': row['response_index'],
                'side': row['side'], 'status': 'complete', 'raw_response': raw,
                'decision': decision, 'parse_status': 'valid' if decision is not None else 'invalid',
                'wall_seconds': time.perf_counter() - started}
            append_jsonl(prediction_path, record)
            records.append(record)
            print(f"PROBE DONE {row['probe_id']} decision={decision}", flush=True)
        write_report(rows, records, args.output_dir, errors)
    return int(len(records) != len(rows))


def score(args):
    rows = load_manifest(args.manifest)
    config_path = args.predictions.parent / 'probe_config.json'
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if config.get('manifest_sha256') != file_hash(args.manifest):
            raise ValueError('Prediction manifest fingerprint does not match')
        rows = selected_rows(rows, config.get('max_pairs', 0))
    # Read-only scoring can inspect an active run without consuming its partial final line.
    metrics = write_report(rows, read_jsonl(args.predictions, complete_only=True), args.output_dir,
                          read_jsonl(args.predictions.parent / 'probe_errors.jsonl', complete_only=True))
    print(json.dumps(metrics['groups']['overall'], ensure_ascii=False))


def execute(args):
    if args.probe_command == 'prepare':
        prepare(args)
    elif args.probe_command == 'run':
        raise SystemExit(run(args))
    else:
        score(args)
