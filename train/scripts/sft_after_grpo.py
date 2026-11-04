#!/usr/bin/env python3
"""Start an independent plain-CE SFT only after a verified GRPO pilot completes."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.files import write_json
from train.scripts.benchmark_after_training import external_weights


def completed_grpo(output, world, steps):
    """No completion from a stale directory, partial rank save, or failed pilot."""
    output = Path(output)
    for rank in range(world):
        path = output / f'memory_report.rank{rank}.json'
        if not path.exists():
            return False
        report = json.loads(path.read_text())
        if (report.get('optimizer_steps') != steps or not report.get('checkpoint_saved')
                or not isinstance(report.get('last_loss'), (int, float))
                or not math.isfinite(report['last_loss'])):
            raise RuntimeError('GRPO did not complete the requested finite updates')
    checkpoint = output / 'checkpoint-final'
    required = [checkpoint / 'training_config.json', checkpoint / 'state/mp_rank_00_model_states.pt']
    required += [checkpoint / f'rng_rank{rank}.pt' for rank in range(world)]
    required += [checkpoint / 'state' / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
                 for rank in range(world)]
    return all(p.is_file() and p.stat().st_size for p in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    from train.src.infrastructure.config import load_config
    sft = load_config(config['sft_config'])
    external_weights(config['sft_output'])
    external_weights(sft['model_path'])
    if sft['decision_loss_mode'] != 'plain_ce' or (Path(sft['model_path']) / 'full_training_model.json').exists():
        raise ValueError('Independent plain-CE SFT must start from the original base model')
    previous = json.loads((Path(config['grpo_prepared']) / 'manifest.json').read_text())
    prepared = json.loads((Path(config['sft_prepared']) / 'manifest.json').read_text())
    if previous['source_sha256'] != prepared['source_sha256'] or previous['videos'] != prepared['videos']:
        raise ValueError('SFT/GRPO video splits differ')
    if args.check_only:
        print(json.dumps(dict(status='validated', gpus=config['gpus'], loss='plain_ce', counts=prepared['counts'])))
        return
    directory = args.config.parent
    lock = (directory / '.sft_after_grpo.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    def mark(stage, **kw):
        write_json(directory / 'sft_after_grpo_status.json', dict(pid=os.getpid(), stage=stage, at=time.time(), **kw))
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        mark('waiting_for_grpo', expected_steps=config['grpo_steps'])
        while True:
            if (directory / 'STOP_SFT').exists():
                mark('cancelled'); return
            status_path = Path(config['grpo_status'])
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('stage') in ('failed', 'stopped'):
                raise RuntimeError('GRPO failed/stopped; automatic SFT will not hide its failure')
            if (status.get('stage') == 'complete' and completed_grpo(config['grpo_output'],
                    len(config['gpus']), config['grpo_steps'])):
                break
            time.sleep(config['poll_seconds'])
        deadline = time.time() + 180
        while True:
            lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            busy = [line for line in lines if line.split(',')[0].strip() in config['gpus']
                    and int(line.split(',')[1]) > 2048]
            if not busy:
                break
            if time.time() > deadline:
                raise RuntimeError('SFT GPUs remain occupied after GRPO; refusing overlap')
            time.sleep(config['poll_seconds'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(config['gpus']),
                   CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONPATH=config['runtime_root'],
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        command = [config['python'], '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(config["gpus"])}', '-m', 'train', '--config', config['sft_config'],
            'run', '--prepared', config['sft_prepared'], '--output', config['sft_output']]
        with (directory / 'plain_ce_sft.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=config['runtime_root'], env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            mark('sft_training', child_pid=child.pid, output=config['sft_output'])
            code = child.wait()
        if code:
            raise RuntimeError(f'Plain-CE SFT exited {code}')
        mark('complete', output=config['sft_output'])
    except KeyboardInterrupt:
        mark('stopped')
    except BaseException as exc:
        mark('failed', error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            from benchmark.runtime.parallel import terminate_owned
            terminate_owned([child])


if __name__ == '__main__':
    main()
