#!/usr/bin/env python3
"""Wait for a complete SFT run, export externally, then run the existing DP suite."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.files import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_weights(path):
    """Check actual storage, allowing a repository symlink to external weights."""
    path = Path(path).resolve()
    forbidden = Path.home() / 'jimodels'
    if path == forbidden or forbidden in path.parents or path == ROOT or ROOT in path.parents:
        raise ValueError('Model weights must be outside ~/jimodels and the repository')
    return path


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def completed_training(config):
    output = Path(config['training_output'])
    run = json.loads((output / 'run_config.json').read_text())
    if sha(output / 'run_config.json') != config['training_run_sha256']:
        raise ValueError('Training run identity changed')
    expected = config['expected_optimizer_steps']
    for rank in range(run['world_size']):
        report = output / f'memory_report.rank{rank}.json'
        if not report.exists():
            raise ValueError(f'Training exited without final report for rank {rank}')
        report = json.loads(report.read_text())
        if not report.get('checkpoint_saved') or report['optimizer_steps'] != expected:
            raise ValueError('Training stopped early; automatic benchmark is not allowed')
    checkpoint = output / 'checkpoint-final'
    if json.loads((checkpoint / 'training_config.json').read_text()) != run['config']:
        raise ValueError('Final checkpoint configuration differs from the selected run')
    required = [checkpoint / f'rng_rank{r}.pt' for r in range(run['world_size'])]
    required += [checkpoint / 'state' / f'bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt'
                 for r in range(run['world_size'])]
    required += [checkpoint / 'state/mp_rank_00_model_states.pt']
    if not all(p.is_file() and p.stat().st_size for p in required):
        raise ValueError('Final checkpoint is incomplete')
    return checkpoint


def prepare(config):
    directory = Path(config['output'])
    directory.mkdir(parents=True, exist_ok=True)
    external_weights(config['export_model'])
    external_weights(config['training_output'])
    if sha(config['data']) != config['data_sha256']:
        raise ValueError('Benchmark data changed')
    data = directory / 'all_qa.json'
    if data.exists() and sha(data) != config['data_sha256']:
        raise ValueError('Benchmark snapshot changed')
    data.write_bytes(Path(config['data']).read_bytes())
    python = config['benchmark_python']
    commands = [
        {'mode': 'realtime', 'adapter': 'minicpmo', 'sequence': 0,
         'argv': [python, '-m', 'benchmark', 'run', 'minicpmo', '--data', str(data),
                  '--videos-dir', config['videos_dir'], '--model-path', config['export_model'],
                  '--output-dir', str(directory / 'suite/realtime/minicpmo'),
                  '--system-prompt', str(ROOT / 'benchmark/configs/prompts/system_prompt.txt'),
                  '--cache-dir', str(ROOT / 'benchmark/cache'),
                  '--agent-call-schema', 'task-start', '--agent-output-protocol', 'task-text']}]
    write_json(directory / 'commands.json', commands)
    suite = [python, str(ROOT / 'benchmark/scripts/run_prepared_suite.py'),
             '--commands', str(directory / 'commands.json'), '--output-dir', str(directory / 'suite'),
             '--gpus', *map(str, config['gpus'])]
    subprocess.run(suite + ['--dry-run'], cwd=ROOT, check=True)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Validate data and DP plans without loading models')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = Path(config['output']); directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.after_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = directory / 'after_training_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
        return
    status = {'pid': os.getpid(), 'status': 'preparing', 'training_output': config['training_output']}
    def mark(state, **extra):
        status.update(status=state, updated_at=time.time(), **extra)
        write_json(status_path, status)
        print(json.dumps(status), flush=True)
    try:
        suite = prepare(config)
        if args.prepare_only:
            mark('prepared')
            return
        mark('waiting_for_training', gpus=config['gpus'])
        while any(alive(pid) for pid in config['training_pids']):
            if (directory / 'STOP').exists():
                mark('cancelled'); return
            time.sleep(config['poll_seconds'])
        if (directory / 'STOP').exists():
            mark('cancelled'); return
        checkpoint = completed_training(config)
        model = external_weights(config['export_model'])
        marker = directory / 'export_complete.json'
        if not marker.exists():
            if model.exists():
                raise ValueError('Unverified export directory exists; refusing to overwrite or reuse it')
            mark('exporting', checkpoint=str(checkpoint), model=str(model))
            command = [config['training_python'], '-m', 'train', '--config', config['training_config'],
                       'export', '--checkpoint', str(checkpoint), '--output', str(model)]
            with (directory / 'export.log').open('ab') as log:
                subprocess.run(command, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(marker, {'model': str(model), 'metadata_sha256': sha(model / 'full_training_model.json')})
        exported = json.loads(marker.read_text())
        if exported['model'] != str(model) or exported['metadata_sha256'] != sha(model / 'full_training_model.json'):
            raise ValueError('Export identity changed')
        mark('benchmark_running')
        with (directory / 'suite.log').open('ab') as log:
            subprocess.run(suite, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        suite_status = json.loads((directory / 'suite/suite_status.json').read_text())
        mark(suite_status['status'])
    except Exception as exc:
        mark('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
