#!/usr/bin/env python3
"""Run an explicitly configured SFT -> realtime evaluation -> judged GRPO chain.

No model is loaded until the SFT completion checks pass. A GPU reservation keeps benchmark schedulers active on the remaining GPUs.
Existing workers finish normally before the judge uses its reserved GPU.
The lease journal and PID start times make cancellation/restart recoverable.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmark/scripts'))
from watch_benchmark_queue import process, read, write, stop_owned


def restore_lease(path):
    journal = read(path)
    if not journal or journal.get('released'):
        return
    reservation = Path(journal['reservation'])
    value = read(reservation)
    if value.get('owner') == journal['owner']:
        write(reservation, {'owner': journal['owner'], 'gpus': []})
    write(path, {**journal, 'released': True})


def acquire_lease(config, path):
    """Reserve only the judge GPU; the suite keeps dispatching on all other GPUs."""
    reservation = Path(config['gpu_reservations'])
    value = read(reservation)
    owner = str(path.resolve())
    if value.get('gpus') and value.get('owner') != owner:
        raise RuntimeError('Another owner has reserved benchmark GPUs')
    journal = dict(owner=owner, reservation=str(reservation),
                   gpus=[read(config['judge_config'])['gpu']])
    write(path, journal)
    write(reservation, {'owner': owner, 'gpus': journal['gpus']})


def verify_split(prepared, sft_prepared):
    """Allow method-specific exclusions, but never move a video across splits."""
    current, source = read(Path(prepared)/'manifest.json'), read(Path(sft_prepared)/'manifest.json')
    if current['source_sha256'] != source['source_sha256']:
        raise ValueError('GRPO and SFT source datasets differ')
    for split in ('train', 'validation'):
        if not set(current['videos'][split]) <= set(source['videos'][split]):
            raise ValueError(f'GRPO changed the SFT {split} video split')
    return current['counts']


def verify_profile(output, world):
    for rank in range(world):
        report = read(Path(output) / f'memory_report.rank{rank}.json')
        if report.get('optimizer_steps') != 1 or not math.isfinite(report.get('last_loss', float('nan'))):
            raise RuntimeError(f'GRPO profile did not complete one finite update on rank {rank}')



def verified_sft_export(config):
    """Explicit GRPO-only entry still requires a completed SFT and committed export."""
    from train.scripts.benchmark_after_training import completed_training,external_weights,sha
    b=read(config['benchmark_config']);checkpoint=completed_training(b)
    model=external_weights(b['export_model']);marker=read(Path(b['output'])/'export_complete.json')
    metadata=model/'full_training_model.json'
    if (marker.get('model')!=str(model) or not metadata.is_file()
            or marker.get('metadata_sha256')!=sha(metadata)):
        raise ValueError('Final SFT export is not verified')
    if Path(read(metadata)['source_checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Export is not from the selected final SFT checkpoint')
    for name in ('grpo_config','profile_config'):
        if Path(read(config[name])['model_path']).resolve()!=model:
            raise ValueError('GRPO/profile model differs from verified SFT export')
    return model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--grpo-only', action='store_true', help='Explicitly skip benchmark dependencies; require the completed SFT export')
    args = parser.parse_args()
    c = read(args.config)
    directory = Path(c['output']); directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.chain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path, lease = directory/'status.json', directory/'gpu_lease.json'
    if read(status_path).get('stage') == 'complete':
        return
    from train.scripts.benchmark_after_training import external_weights
    for key in ('profile_output', 'grpo_output'):
        external_weights(c[key])
    if set(c['train_gpus']) & {read(c['judge_config'])['gpu']}:
        raise ValueError('Judge GPU overlaps training GPUs')
    counts = verify_split(c['prepared'], c['sft_prepared'])
    if args.grpo_only:
        verified_sft_export(c)
    if args.check_only:
        print(json.dumps(dict(status='validated', counts=counts, train_gpus=c['train_gpus'],
                              judge_gpu=read(c['judge_config'])['gpu'])))
        return
    children = []
    def mark(stage, **extra):
        write(status_path, dict(pid=os.getpid(), stage=stage, at=time.time(), **extra))
        print(json.dumps(read(status_path)), flush=True)
    def check_stop():
        if (directory/'STOP').exists():
            raise InterruptedError('STOP requested')
    def wait_for(predicate, stage):
        mark(stage)
        while not predicate():
            check_stop(); time.sleep(c['poll_seconds'])
        check_stop()
    def start(argv, name, env=None):
        log = (directory/f'{name}.log').open('ab')
        try:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        children.append(child)
        return child
    def run(argv, name, env=None):
        mark(name)
        child = start(argv, name, env)
        while child.poll() is None:
            check_stop(); time.sleep(c['poll_seconds'])
        if child.returncode:
            raise RuntimeError(f'{name} exited {child.returncode}; see {directory/name}.log')
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if read(lease) and not read(lease).get('released'):
            restore_lease(lease)
        if args.grpo_only:
            mark('benchmarks_skipped', reason='Explicit --grpo-only request', model=str(verified_sft_export(c)))
        else:
            run([c['python'], str(ROOT/'train/scripts/benchmark_after_training.py'),
                 '--config', c['benchmark_config']], 'waiting_sft_then_realtime_benchmark')
            b = read(c['benchmark_config'])
            if read(Path(b['output'])/'after_training_status.json').get('status') != 'complete':
                raise RuntimeError('Realtime benchmark did not finish successfully; GRPO not started')
            # OmniPro also evaluates this final SFT; let all its inference and judge jobs finish.
            wait_for(lambda: read(c['omnipro_status']).get('stage') == 'complete', 'waiting_for_omnipro')
        acquire_lease(c, lease)
        gpu = read(c['judge_config'])['gpu']
        def gpu_free():
            used = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            return int(used.strip()) <= read(c['judge_config'])['max_occupied_memory_mib']
        wait_for(gpu_free, 'waiting_for_judge_gpu_existing_worker')
        judge = start([c['python'], str(ROOT/'train/scripts/serve_grpo_judge.py'),
                       '--config', c['judge_config'], '--start'], 'judge')
        deadline = time.time()+c['judge_startup_timeout_seconds']
        def ready():
            if judge.poll() is not None or time.time() > deadline:
                raise RuntimeError('Judge exited or startup timed out')
            try:
                with urllib.request.urlopen(c['judge_health_url'], timeout=5) as response:
                    return response.status == 200
            except OSError:
                return False
        wait_for(ready, 'starting_judge')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, c['train_gpus'])),
               'PYTHONPATH': str(ROOT), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        # Read secrets only into the subprocess environment, never into config/status/logs.
        from dotenv import dotenv_values
        env.update({k:v for k,v in dotenv_values(c['env_file']).items() if v is not None and k.startswith('WANDB_')})
        prefix = [c['python'], '-m', 'torch.distributed.run', '--standalone',
                  f'--nproc_per_node={len(c["train_gpus"])}', '-m', 'train', '--config']
        if not read(directory/'profile_complete.json'):
            run(prefix+[c['profile_config'], 'profile', '--prepared', c['prepared'],
                        '--output', c['profile_output']], 'grpo_profile', env)
            verify_profile(c['profile_output'], len(c['train_gpus']))
            write(directory/'profile_complete.json', dict(output=c['profile_output']))
        grpo_command = prefix+[c['grpo_config'], 'grpo', '--prepared', c['prepared'],
                               '--output', c['grpo_output']]
        if c.get('grpo_max_steps'):
            grpo_command += ['--max-steps', str(c['grpo_max_steps'])]
        run(grpo_command, 'grpo_training', env)
        mark('complete')
    except BaseException as exc:
        mark('stopped' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        raise
    finally:
        # Stop only process groups launched by this controller, then release the judge GPU reservation.
        for child in reversed(children):
            if child.poll() is None:
                entry = process(child.pid)
                if entry:
                    stop_owned([entry], 30)
                child.wait()
        restore_lease(lease)


if __name__ == '__main__':
    main()
