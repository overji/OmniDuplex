#!/usr/bin/env python3
"""Run an isolated reference-history, negative-only Probe diagnostic on fixed DP shards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.files import read_jsonl, write_json, atomic_write_text
from benchmark.probe.data import load_manifest, reference_history, file_hash
from benchmark.probe.metrics import write_report
from benchmark.runtime.parallel import terminate_owned
from benchmark.scripts.run_prepared_suite import build_plan, option


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'manifest', 'model-path', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--worker-python', default=sys.executable)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--active-queue', type=Path)
    parser.add_argument('--next-suite-status', type=Path)
    args = parser.parse_args()
    if args.poll_seconds < 1 or len(set(args.gpus)) != len(args.gpus):
        parser.error('Require a positive poll interval and distinct GPUs')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.history_trial.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows, excluded = reference_history(load_manifest(args.manifest), json.loads(args.data.read_text()),
                                        file_hash(args.data))
    negative = [r for r in rows if r['side']=='pre']
    if not negative:
        raise ValueError('No history-safe negative probes')
    write_json(output/'history_exclusions.json', excluded)
    command = [args.worker_python, '-m', 'benchmark', 'probe', 'run', 'minicpmo',
        '--manifest', str(args.manifest.resolve()), '--model-path', str(args.model_path.resolve()),
        '--output-protocol', 'reply-task-text', '--prompt-layout', 'task-specific',
        '--max-new-tokens', str(args.max_new_tokens), '--history-mode', 'reference',
        '--history-data', str(args.data.resolve()), '--probe-side', 'pre', '--skip-failed']
    task, = build_plan([dict(mode='probe', adapter='minicpmo', argv=command)],
                       args.gpus, output)
    workers = task['commands']
    for rank, worker in enumerate(workers):
        worker['argv'] = option(worker['argv'], '--output-dir', output/f'worker_{rank}')
    smoke_pair = next((r['pair_id'] for r in negative if r['handled_count']>0), negative[0]['pair_id'])
    smoke_manifest = output/'history_smoke_manifest.jsonl'
    atomic_write_text(smoke_manifest, ''.join(json.dumps(r, ensure_ascii=False)+'\n'
                                            for r in rows if r['pair_id']==smoke_pair))
    smoke = option(option(command, '--manifest', smoke_manifest), '--output-dir', output/'smoke')
    plan = dict(gpus=args.gpus, workers=workers, smoke=smoke, expected_negative_probes=len(negative),
                history_mode='reference', probe_side='pre', failure_policy='skip')
    path = output/'history_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('History diagnostic plan changed; use a new output directory')
    write_json(path, plan)
    status = dict(pid=os.getpid(),status='running',gpus=args.gpus,workers=[],failure_policy='skip',jobs=[])
    processes, logs = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def save():
        write_json(output/'parallel_status.json',status)
    def launch(argv, gpus, logpath):
        log=logpath.open('ab');logs.append(log)
        child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(gpus),
            'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID'},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(child)
        return child
    def collect():
        predictions, errors = [], []
        for rank in range(len(workers)):
            directory=output/f'worker_{rank}'
            predictions+=read_jsonl(directory/'probe_predictions.jsonl',complete_only=True)
            errors+=read_jsonl(directory/'probe_errors.jsonl',complete_only=True)
        atomic_write_text(output/'probe_predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
        atomic_write_text(output/'probe_errors.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors))
        write_report(negative,predictions,output,errors)
        status['completed_probes']=len(predictions)
        status['jobs']=[dict(mode='probe',adapter='minicpmo_gt_history_negative',output=str(output),
            status=status['status'],pids=[c.pid for c in processes],completed_probes=len(predictions),
            exit_codes=[c.poll() for c in processes])]
    try:
        save()
        child=launch(smoke,[args.gpus[0]],output/'smoke.log')
        code=child.wait();processes.remove(child)
        if code:
            raise RuntimeError('Reference-history negative smoke failed; see smoke.log')
        for rank,worker in enumerate(workers):
            child=launch(worker['argv'],worker['gpus'],output/f'worker_{rank}.log')
            status['workers'].append(dict(rank=rank,pid=child.pid,exit_code=None))
        save()
        while any(child.poll() is None for child in processes):
            for worker,child in zip(status['workers'],processes):worker['exit_code']=child.poll()
            collect();save();time.sleep(args.poll_seconds)
        for worker,child in zip(status['workers'],processes):worker['exit_code']=child.returncode
        status['status']='complete'
        status['has_failures']=any(child.returncode for child in processes)
        collect();save()
        if args.active_queue and args.next_suite_status:
            active=json.loads(args.active_queue.read_text())
            active['next_suite_status']=str(args.next_suite_status.resolve())
            write_json(args.active_queue,active)
        return 0  # Final per-request failures remain coverage failures, never block the next suite.
    except KeyboardInterrupt:
        status['status']='interrupted';save();return 1
    except Exception:
        status['status']='failed';save();raise
    finally:
        terminate_owned(processes)
        for log in logs:log.close()
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
