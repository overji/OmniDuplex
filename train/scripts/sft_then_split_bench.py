#!/usr/bin/env python3
"""Train on all configured GPUs, then evaluate AOI and OmniPro on disjoint pools."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.files import write_json
from train.scripts.benchmark_after_training import completed_training, external_weights


def previous_ready(config):
    """A terminal per-item Judge failure is allowed; a crashed service is not.

    Never signal the previous queue. PID start times prevent a reused PID from
    being mistaken for the benchmark owner whose exit we are waiting for.
    """
    def read(name):
        path = Path(config[name])
        return json.loads(path.read_text()) if path.exists() else {}
    chain, omni, aoi, judge = (read(name) for name in ('chain_status', 'omnipro_status', 'aoi_status', 'judge_stage'))
    if chain.get('stage') in ('failed', 'stopped'):
        raise RuntimeError('Previous benchmark chain failed/stopped; inspect it before starting new training')
    if chain.get('stage') not in ('complete', 'benchmark_failed'):
        return False
    if omni.get('stage') != 'complete' or judge.get('status') not in ('complete', 'incomplete'):
        raise RuntimeError('Previous four-GPU Judge did not reach a completed terminal stage')
    if aoi.get('status') not in ('complete', 'incomplete'):
        raise RuntimeError('Previous AOI inference did not finish normally')
    from benchmark.scripts.watch_benchmark_queue import process
    return not any((current := process(item['pid'])) and current['start'] == item['start']
                   for item in config['processes'])


def validate(config):
    from train.src.infrastructure.config import load_config
    training = load_config(config['training_config'])
    for path, expected in config.get('input_hashes', {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input changed: ' + path)
    if config.get('video_metadata'):
        metadata = json.loads(Path(config['video_metadata']).read_text())
        for row in metadata['videos']:
            video = Path(training['videos_dir']) / row['file']
            stat = video.stat()
            if stat.st_size != row['size_bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                raise ValueError('Video changed after duration audit: ' + str(video))
    aoi, omni = config['aoi_gpus'], config['omnipro_gpus']
    if (len(set(config['train_gpus'])) != len(config['train_gpus']) or set(aoi) & set(omni)
            or set(aoi) | set(omni) != set(config['train_gpus'])):
        raise ValueError('Benchmark pools must partition the training GPUs')
    if training['decision_loss_mode'] != 'plain_ce' or training['parallel_mode'] != 'data_parallel':
        raise ValueError('This comparison requires independent-QA ordinary CE SFT')
    prepared = Path(config['prepared'])
    manifest = json.loads((prepared / 'manifest.json').read_text())
    records = json.loads((prepared / 'train.json').read_text())
    if set(manifest['videos']['train']) & set(manifest['videos']['validation']):
        raise ValueError('Training and validation videos overlap')
    batch = len(config['train_gpus']) * training['gradient_accumulation_steps']
    steps = math.ceil(len(records) / batch) * training['epochs']
    if steps != config['expected_steps']:
        raise ValueError('Expected training steps differ from the configured batch')
    for name in ('training_output', 'export_model'):
        external_weights(config[name])
    if config.get('profile_output'):
        external_weights(config['profile_output'])
    if config.get('runtime_root'):
        for relative in ('benchmark/scripts/watch_live_results.py', 'benchmark/src/evaluation/live_report.py'):
            if not (Path(config['runtime_root']) / relative).is_file():
                raise ValueError('Frozen runtime is missing reporting component: ' + relative)
    external_weights(training['model_path'])
    return dict(global_batch_size=batch, expected_steps=steps, counts=manifest['counts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--omnipro-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verified = validate(config)
    if args.check_only:
        print(json.dumps(dict(status='validated', **verified)))
        return
    directory = args.config.parent
    if args.omnipro_only:
        # Reuse inference, bounded monitoring and four-replica Judge, without
        # the old controller's pause/resume of unrelated benchmark queues.
        from benchmark.runtime.omnipro_queue import Queue
        queue = Queue(config['omnipro_config'])
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            with (queue.root / '.omnipro.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                queue.run_models(queue.config['models'])
                queue.state('complete')
        finally:
            queue.stop_children()
        return
    lock = (directory / '.chain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children = []
    def mark(stage, **extra):
        write_json(directory / 'status.json', dict(pid=os.getpid(), stage=stage, at=time.time(), **extra))
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def launch(argv, name, env):
        with (directory / f'{name}.log').open('ab') as log:
            child = subprocess.Popen(argv, cwd=config['runtime_root'], env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    try:
        if Path(config['training_output']).exists() or Path(config['export_model']).exists():
            raise ValueError('Use fresh training/export directories; never overwrite checkpoints')
        if config.get('wait_for_previous'):
            mark('waiting_for_previous_benchmarks_and_judge')
            while not previous_ready(config['wait_for_previous']):
                if (directory / 'STOP').exists():
                    raise KeyboardInterrupt
                time.sleep(config['poll_seconds'])
            validate(config)  # Recheck immutable inputs after a potentially long wait.
        release_deadline = time.time() + config.get('gpu_release_timeout_seconds', 0)
        while True:
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()
            occupied = any(line.split(',')[0].strip() in config['train_gpus']
                           and int(line.split(',')[1]) > 2048 for line in used)
            if not occupied:
                break
            if time.time() >= release_deadline:
                raise RuntimeError('Training GPUs are occupied; refusing overlap')
            mark('waiting_for_gpu_release')
            if (directory / 'STOP').exists():
                raise KeyboardInterrupt
            time.sleep(config['poll_seconds'])
        env = dict(os.environ, PYTHONPATH=config['runtime_root'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=','.join(config['train_gpus']), PYTHONUNBUFFERED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if config.get('profile_config'):
            if Path(config['profile_output']).exists():
                raise ValueError('Use a fresh profile output directory')
            profile = launch([config['python'], '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={len(config["train_gpus"])}', '-m', 'train', '--config', config['profile_config'],
                'profile', '--prepared', config['prepared'], '--output', config['profile_output']], 'profile', env)
            mark('eight_gpu_profile', child_pid=profile.pid)
            available_min = float('inf')
            while profile.poll() is None:
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) / 2**20
                available_min = min(available_min, available)
                write_json(directory / 'profile_memory.json', dict(minimum_host_available_gib=available_min, at=time.time()))
                if available < config['min_host_available_gib']:
                    raise RuntimeError('Profile host memory reserve exhausted; no automatic parameter changes')
                if (directory / 'STOP').exists():
                    raise KeyboardInterrupt
                time.sleep(config['poll_seconds'])
            if profile.returncode:
                raise RuntimeError('Eight-GPU memory/profile test failed; full training will not start')
            from train.scripts.sft_bench_grpo import verify_profile
            verify_profile(config['profile_output'], len(config['train_gpus']))
            for rank in range(len(config['train_gpus'])):
                report = json.loads((Path(config['profile_output']) / f'memory_report.rank{rank}.json').read_text())
                if report.get('excluded') or report.get('checkpoint_saved'):
                    raise RuntimeError('Profile excluded a selected QA or unexpectedly saved weights')
            write_json(directory / 'profile_complete.json', dict(output=config['profile_output'],
                       minimum_host_available_gib=available_min))
            children.remove(profile)
        child = launch([config['python'], '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(config["train_gpus"])}', '-m', 'train', '--config', config['training_config'],
            'run', '--prepared', config['prepared'], '--output', config['training_output']], 'training', env)
        mark('plain_ce_training', child_pid=child.pid, **verified)
        if child.wait():
            raise RuntimeError('Plain-CE SFT failed; benchmarks will not start')
        children.remove(child)
        run_config = Path(config['training_output']) / 'run_config.json'
        benchmark_path = Path(config['aoi_config'])
        benchmark = json.loads(benchmark_path.read_text())
        benchmark['training_run_sha256'] = hashlib.sha256(run_config.read_bytes()).hexdigest()
        write_json(benchmark_path, benchmark)
        completed_training(benchmark)  # requires final reports and all eight checkpoint shards
        env['CUDA_VISIBLE_DEVICES'] = ''
        aoi = launch([config['python'], str(ROOT / 'train/scripts/benchmark_after_training.py'),
                      '--config', str(benchmark_path)], 'aoi_export_and_benchmark', env)
        mark('exporting_for_benchmarks', aoi_pid=aoi.pid)
        marker = Path(benchmark['output']) / 'export_complete.json'
        while not marker.exists():
            if aoi.poll() is not None:
                raise RuntimeError('Export/AOI job exited before its export completion marker')
            time.sleep(config['poll_seconds'])
        record = json.loads(marker.read_text())
        metadata = Path(config['export_model']) / 'full_training_model.json'
        if (Path(record['model']).resolve() != Path(config['export_model']).resolve()
                or record['metadata_sha256'] != hashlib.sha256(metadata.read_bytes()).hexdigest()
                or Path(json.loads(metadata.read_text())['source_checkpoint']).resolve()
                != (Path(config['training_output']) / 'checkpoint-final').resolve()):
            raise RuntimeError('Final export identity mismatch')
        omni = launch([config['python'], str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
                       '--omnipro-only'], 'omnipro_benchmark', env)
        watcher = launch([config['python'], str(ROOT / 'benchmark/scripts/watch_live_results.py'),
                          '--config', config['live_reports']], 'live_results', env)
        mark('benchmarks_running', aoi_pid=aoi.pid, omnipro_pid=omni.pid, watcher_pid=watcher.pid)
        codes = [aoi.wait(), omni.wait()]
        if codes == [0, 0] and config.get('content_judge_config'):
            judge = launch([config['python'], str(ROOT / 'benchmark/scripts/run_local_content_judge.py'),
                            '--config', config['content_judge_config']], 'aoi_content_judge', env)
            mark('aoi_content_judging', judge_pid=judge.pid)
            if judge.wait():
                raise RuntimeError('AOI content Judge failed; subsequent jobs must wait')
            children.remove(judge)
        # Commit a final report after both model pools have finished.
        from benchmark.runtime.parallel import terminate_owned
        terminate_owned([watcher])
        subprocess.run([config['python'], str(ROOT / 'benchmark/scripts/watch_live_results.py'),
                        '--config', config['live_reports'], '--once'], cwd=config['runtime_root'], env=env, check=True)
        mark('complete' if codes == [0, 0] else 'benchmark_failed', exit_codes=codes)
    except KeyboardInterrupt:
        mark('stopped')
    except BaseException as exc:
        mark('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        from benchmark.runtime.parallel import terminate_owned
        terminate_owned(children)


if __name__ == '__main__':
    main()
