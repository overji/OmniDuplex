"""Fixed-seed diagnostic trajectories and unbiased reuse of the first rollout.

This is evaluation-only data, not a GRPO training corpus. Each candidate uses the
normal streaming runner; changing only its seed preserves the evaluation protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from benchmark.cli.main import build_parser
from benchmark.data.files import append_jsonl, read_jsonl, write_json
from benchmark.runtime.parallel import prepare_plan


def option(argv, name, value):
    argv = list(argv)
    if name in argv:
        argv[argv.index(name) + 1] = str(value)
    else:
        argv += [name, str(value)]
    return argv


def parse_parallel(argv):
    """Use the production CLI defaults and DP shard assignment unchanged."""
    offset = argv.index('benchmark') + 1
    args = build_parser().parse_args(argv[offset:])
    if vars(args).pop('command') != 'parallel' or vars(args).pop('adapter_name') != 'minicpmo':
        raise ValueError('Diagnostic rollouts currently require parallel minicpmo')
    controls = {key: vars(args).pop(key) for key in ('gpus', 'worker_python', 'poll_seconds', 'sample_indices')}
    controls['worker_python'] = controls['worker_python'] or argv[0]
    return args, controls


def prepare(config):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    args, controls = parse_parallel(config['full_command'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_start != 0 or args.sample_step != 1 or args.max_samples:
        raise ValueError('Use sample-indices, rather than positional limits, for staged evaluation')
    samples = json.loads(args.data.read_text())
    available = (json.loads(Path(controls['sample_indices']).read_text())
                 if controls['sample_indices'] else list(range(len(samples))))
    # prepare_plan performs the normal duplicate/range checks before sampling.
    _, full_plan = prepare_plan(args, 'minicpmo', controls['gpus'], controls['worker_python'],
                               sample_indices=available)
    eligible = sorted(i for shard in full_plan['assignments'] for i in shard)
    count, seeds = config['sample_count'], config['rollout_seeds']
    if type(count) is not int or not 0 < count <= len(eligible):
        raise ValueError('sample_count must fit the eligible QA set')
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or seeds[0] != args.seed):
        raise ValueError('Unique rollout seeds must start with the official evaluation seed')
    if args.decode_mode != 'sampling' or args.max_seconds:
        raise ValueError('Diagnostics require sampling and full-video trajectories')
    selected = random.Random(config['selection_seed']).sample(eligible, count)
    manifest = dict(version=1, purpose='diagnostic_eval_only', full_command=config['full_command'],
        selection_seed=config['selection_seed'], rollout_seeds=seeds, sample_indices=selected,
        official_rollout=0, data_sha256=full_plan['data_sha256'], prompt_sha256=full_plan['prompt_sha256'],
        samples=[dict(sample_index=i, qa_sha256=hashlib.sha256(json.dumps(samples[i], sort_keys=True,
            ensure_ascii=False).encode()).hexdigest(), task=samples[i]['task'], video=samples[i]['video'])
                 for i in selected])
    path = root / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Diagnostic manifest changed; use a new output directory')
    write_json(path, manifest)
    indices = root / 'sample_indices.json'
    write_json(indices, selected)
    commands = []
    for number, seed in enumerate(seeds):
        command = option(config['full_command'], '--output-dir', root / f'rollout_{number:02d}')
        command = option(option(command, '--sample-indices', indices), '--seed', seed)
        commands.append(command)
    write_json(root / 'commands.json', commands)
    return manifest, commands


def completed_rows(root, indices):
    rows = read_jsonl(root / 'predictions.jsonl')
    if (len(rows) != len(indices) or {r['sample_index'] for r in rows} != set(indices)
            or any(r['status'] != 'complete' for r in rows)):
        raise ValueError('Rollout contains missing, duplicate or partial trajectories: ' + str(root))
    return rows


def adopt_first(config, manifest):
    """Seed the official DP workers without selecting trajectories by their scores.

    GPU placement and output directories may change; every inference parameter,
    dependency version, input hash and protocol must match the source worker.
    Raw chunks and rendered prompts follow their original sample indices.
    """
    from benchmark.adapters import minicpmo
    from benchmark.data.agent_inputs import input_protocol
    root = Path(config['output'])
    source = root / 'rollout_00'
    completed_rows(source, manifest['sample_indices'])
    args, controls = parse_parallel(config['full_command'])
    plan = json.loads((args.output_dir / 'parallel_plan.json').read_text())
    source_workers = {}
    for directory in source.glob('worker_*'):
        for row in read_jsonl(directory / 'predictions.jsonl'):
            index = row['sample_index']
            if index in source_workers:
                raise ValueError('Duplicate rollout worker prediction')
            source_workers[index] = (directory, row)
    for rank, assigned in enumerate(plan['assignments']):
        target = args.output_dir / f'worker_{rank}'
        target.mkdir(exist_ok=True)
        target_args = argparse.Namespace(**vars(args))
        target_args.output_dir = target
        target_args.device = 'cuda:0'
        minicpmo.prepare_args(target_args)
        old_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(controls['gpus'][rank])
            expected = minicpmo.run_config(target_args, args.system_prompt.read_text())
        finally:
            if old_gpu is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = old_gpu
        expected.update(agent_input_protocol=input_protocol(target_args),
                        agent_call_schema=target_args.agent_call_schema)
        previous = target / 'run_config.json'
        if previous.exists() and json.loads(previous.read_text()) != expected:
            raise ValueError('Official worker configuration changed')
        write_json(previous, expected)
        existing = {r['sample_index']: r for r in read_jsonl(target / 'predictions.jsonl')}
        for index in assigned:
            if index not in manifest['sample_indices']:
                continue
            directory, row = source_workers[index]
            source_config = json.loads((directory / 'run_config.json').read_text())
            ignored = {'output_dir', 'cuda_visible_devices'}
            if ({k: v for k, v in source_config.items() if k not in ignored}
                    != {k: v for k, v in expected.items() if k not in ignored}):
                raise ValueError('Seed-zero inference configuration differs from official run')
            if index in existing:
                if existing[index] != row:
                    raise ValueError('Conflicting official prediction; refusing overwrite')
                continue
            raw = directory / f'sample_{index:03d}.chunks.jsonl'
            if not raw.is_file():
                raise ValueError('Missing raw trajectory: ' + str(raw))
            shutil.copyfile(raw, target / raw.name)
            prompts = [p for p in read_jsonl(directory / 'sample_system_prompts.jsonl')
                       if p['sample_index'] == index]
            if not prompts or hashlib.sha256(prompts[-1]['system_prompt'].encode()).hexdigest() != row['system_prompt_sha256']:
                raise ValueError('Missing or changed rendered prompt')
            append_jsonl(target / 'sample_system_prompts.jsonl', prompts[-1])
            append_jsonl(target / 'predictions.jsonl', row)
    write_json(args.output_dir / 'rollout_reuse.json', dict(
        source=str(source), official_rollout=0, sample_indices=manifest['sample_indices'],
        selection='fixed_first_seed_no_best_of_n', manifest_sha256=hashlib.sha256(
            (root / 'manifest.json').read_bytes()).hexdigest()))


def summarize(config, manifest):
    root = Path(config['output'])
    records = []
    for number, seed in enumerate(manifest['rollout_seeds']):
        directory = root / f'rollout_{number:02d}'
        rows = completed_rows(directory, manifest['sample_indices'])
        scores = {r['sample_index']: r for r in json.loads((directory / 'metrics.json').read_text())['per_sample']}
        for row in rows:
            records.append(dict(rollout_index=number, seed=seed, sample_seed=seed + row['sample_index'],
                sample_index=row['sample_index'], prediction=row, timing=scores[row['sample_index']],
                source=str(directory)))
    # A single portable index of all predictions; original per-second chunks stay in workers.
    from benchmark.data.files import atomic_write_text
    atomic_write_text(root / 'rollouts.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    write_json(root / 'complete.json', dict(qa_count=len(manifest['sample_indices']),
        trajectories=len(records), official_rollout=0, purpose='diagnostic_eval_only'))
