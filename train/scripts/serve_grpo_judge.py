#!/usr/bin/env python3
"""Inspect a local judge launch by default; --start explicitly replaces this process."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def command(config):
    model = Path(config['model_path']).resolve()
    forbidden = Path.home() / 'jimodels'
    if model == forbidden or forbidden in model.parents:
        raise ValueError('Judge weights must be physically outside ~/jimodels')
    if config['host'] != '127.0.0.1':
        raise ValueError('The training judge must listen on localhost only')
    python = Path(config['python'])
    python = python if python.is_absolute() else ROOT / python
    return [str(python), '-m', 'vllm.entrypoints.openai.api_server', '--model', str(model),
            '--served-model-name', config['served_model_name'], '--host', config['host'],
            '--port', str(config['port']), '--dtype', config['dtype'],
            '--max-model-len', str(config['max_model_len']), '--max-num-seqs', str(config['max_num_seqs']),
            '--gpu-memory-utilization', str(config['gpu_memory_utilization']), '--tensor-parallel-size', '1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'train/configs/judge_qwen27b_service.json')
    parser.add_argument('--start', action='store_true', help='Start in foreground; never stops other GPU processes')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    argv = command(config)
    print(json.dumps({'gpu': config['gpu'], 'command': argv, 'start_requested': args.start}), flush=True)
    if not args.start:
        return
    occupied = subprocess.check_output(['nvidia-smi', f'--id={config["gpu"]}',
                                       '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(occupied.strip()) > config['max_occupied_memory_mib']:
        raise RuntimeError('Judge GPU is occupied; explicitly migrate the benchmark before starting the judge')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(config['gpu']), 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1'}
    os.execve(argv[0], argv, env)


if __name__ == '__main__':
    main()
