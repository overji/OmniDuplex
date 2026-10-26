#!/usr/bin/env python3
"""Run a prepared suite with fixed DP shards and supervised, resumable outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.data.files import read_jsonl, write_json, atomic_write_text
from benchmark.probe.data import load_manifest
from benchmark.probe.metrics import write_report
from benchmark.runtime.parallel import terminate_owned


def option(argv, name, value):
    argv = list(argv)
    if name in argv:
        argv[argv.index(name) + 1] = str(value)
    else:
        argv += [name, str(value)]
    return argv


def build_plan(commands, gpus, output, reuse_root=None):
    tasks = []
    # Start with the previously exercised native streaming adapter.
    order = {'minicpmo': 0, 'gander': 1, 'livestar': 2, 'joyai': 3}
    if any('sequence' in entry for entry in commands):
        sequences = [entry['sequence'] for entry in commands]
        if any(type(n) is not int for n in sequences) or len(set(sequences)) != len(sequences):
            raise ValueError('Scheduled commands require distinct integer sequence values')
        commands = sorted(commands, key=lambda c: c['sequence'])
    else:
        commands = sorted(commands, key=lambda c: (c['mode'] != 'realtime', order.get(c['adapter'], 4)))
    for entry in commands:
        mode, name = entry['mode'], entry['adapter']
        task_gpus = entry.get('gpus', gpus)
        if not task_gpus or len(set(task_gpus)) != len(task_gpus) or not set(task_gpus) <= set(gpus):
            raise ValueError('Task GPUs must be a nonempty distinct subset of suite GPUs')
        variant = entry.get('variant')
        label = f'{name}_{variant}' if variant else name
        if Path(label).name != label or label in ('.', '..'):
            raise ValueError('Unsafe task label')
        directory = output / mode / label
        reuse = reuse_root and mode == 'realtime' and (reuse_root / mode / name / 'parallel_plan.json').exists()
        if variant:
            reuse = False
        if reuse:
            directory = reuse_root / mode / name
        command = option(entry['argv'], '--output-dir', directory)
        if mode == 'realtime':
            command[command.index('run')] = 'parallel'
            command += ['--gpus', *task_gpus]
            if '--sample-indices' not in command:
                command += ['--sample-indices', str((reuse_root if reuse else output) / 'available_sample_indices.json')]
            if name != 'gander' and '--agent-output-protocol' not in command:
                command += ['--agent-output-protocol', 'structured']
            if name == 'joyai' and '--joyai-services' not in command:
                command += ['--joyai-services', str(REPO / 'benchmark/probes/full_2110/joyai_services.json')]
            # MiniCPM uses an already exercised adapter; new adapters first run one full video.
            smoke = None
            if (name != 'minicpmo' or variant) and not entry.get('skip_smoke', False):
                smoke = list(command)
                start = smoke.index('--gpus')
                del smoke[start + 2:start + 1 + len(task_gpus)]
                smoke = option(smoke, '--output-dir', output / 'smoke' / mode / label)
                if entry.get('smoke_sample_indices'):
                    smoke = option(smoke, '--sample-indices', entry['smoke_sample_indices'])
                smoke += ['--max-samples', '1']
            tasks.append(dict(mode=mode, adapter=name, smoke=smoke, commands=[command], output=str(directory)))
        else:
            manifest = Path(command[command.index('--manifest') + 1])
            rows = load_manifest(manifest)
            pairs = list(dict.fromkeys(r['pair_id'] for r in rows))
            # The complete Qwen3-Omni checkpoint needs two 80 GB GPUs per replica.
            width = 2 if name == 'qwen3_omni' else 1
            groups = [task_gpus[i:i+width] for i in range(0, len(task_gpus), width)]
            if any(len(group) != width for group in groups):
                raise ValueError('GPU count must be divisible by replica width')
            workers = []
            for rank, group in enumerate(groups):
                selected = set(pairs[rank::len(groups)])
                shard = output / 'manifests' / f'{label}_{rank}.jsonl'
                shard.parent.mkdir(parents=True, exist_ok=True)
                content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if r['pair_id'] in selected)
                if shard.exists() and shard.read_text() != content:
                    raise ValueError('Existing shard changed')
                atomic_write_text(shard, content)
                cmd = option(command, '--manifest', shard)
                cmd = option(cmd, '--output-dir', directory / f'worker_{rank}')
                workers.append({'argv': cmd, 'gpus': group})
            smoke = option(command, '--output-dir', output / 'smoke' / mode / label) + ['--max-pairs', '1']
            if entry.get('smoke_agent'):
                first = next((row['pair_id'] for row in rows if row['mode'] == 'agent'), None)
                if first is None:
                    raise ValueError('Agent smoke requested but manifest has no Agent pairs')
                path = output / 'manifests' / f'{label}_agent_smoke.jsonl'
                content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows if row['pair_id'] == first)
                if path.exists() and path.read_text() != content:
                    raise ValueError('Existing smoke manifest changed')
                atomic_write_text(path, content)
                smoke = option(smoke, '--manifest', path)
            tasks.append(dict(mode=mode, adapter=name, smoke=smoke, smoke_gpus=groups[0],
                              commands=workers, output=str(directory), manifest=str(manifest)))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--probe-retry-errors-once', action='store_true',
                   help='Retry probe failures once after the first pass, then continue with partial results')
    p.add_argument('--probe-skip-failed', action='store_true',
                   help='Do not retry failed probe requests; keep final partial tasks and continue')
    p.add_argument('--reuse-root', type=Path, help='Resume existing realtime shards without changing their plans')
    p.add_argument('--joyai-services', type=Path, help='Explicit compatible JoyAI service configuration')
    args = p.parse_args()
    if args.probe_retry_errors_once and args.probe_skip_failed:
        p.error('Choose one probe error policy')
    if len(set(args.gpus)) != len(args.gpus):
        p.error('GPU indices must be distinct')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commands = json.loads(args.commands.read_text())
    source = next((c['argv'] for c in commands if '--data' in c['argv'] and '--videos-dir' in c['argv']), None)
    if source is None:
        p.error('Suite needs one realtime entry supplying --data and --videos-dir')
    data = Path(source[source.index('--data') + 1])
    videos = Path(source[source.index('--videos-dir') + 1])
    samples = json.loads(data.read_text())
    available = [i for i, sample in enumerate(samples) if (videos / sample['video']).is_file()]
    write_json(output / 'available_sample_indices.json', available)
    write_json(output / 'excluded_samples.json', [dict(sample_index=i, video=s['video'], reason='missing_video')
               for i, s in enumerate(samples) if i not in available])
    tasks = build_plan(commands, args.gpus, output, args.reuse_root.resolve() if args.reuse_root else None)
    if args.joyai_services:
        for task in tasks:
            if task['adapter'] == 'joyai':
                task['commands'] = [option(command, '--joyai-services', args.joyai_services.resolve()) for command in task['commands']]
                if task['smoke']:
                    task['smoke'] = option(task['smoke'], '--joyai-services', args.joyai_services.resolve())
    plan = {'gpus': args.gpus, 'source_samples': len(samples), 'available_samples': len(available), 'tasks': tasks}
    plan_path = output / 'suite_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Suite plan changed; use a new output directory')
    write_json(plan_path, plan)
    if args.dry_run:
        print(f'Prepared {len(tasks)} model/mode jobs; no inference started')
        return
    previous_status_path = output / 'suite_status.json'
    previous_status = json.loads(previous_status_path.read_text()) if previous_status_path.exists() else {}
    processes, logs = [], []
    status = {'pid': os.getpid(), 'status': 'running', 'gpus': args.gpus, 'jobs': []}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def save():
        write_json(output / 'suite_status.json', status)
    def launch(command, path, gpus=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open('ab'); logs.append(log)
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
               'HF_HUB_OFFLINE': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
               'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        if gpus:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    try:
        save()
        for task in tasks:
            item = {'mode': task['mode'], 'adapter': task['adapter'], 'status': 'starting', 'output': task['output']}
            status['jobs'].append(item); save()
            print(f"START {task['mode']}/{task['adapter']}", flush=True)
            directory = Path(task['output'])
            previous = next((j for j in previous_status.get('jobs', [])
                             if j['output'] == task['output']), None)
            if (args.probe_skip_failed and task['mode'] == 'probe' and previous
                    and previous['status'] in ('complete', 'incomplete_final')):
                item.update(previous, pids=[], exit_codes=[], reused=True)
                save()
                continue
            existing = directory / 'parallel_status.json'
            if task['mode'] == 'realtime' and existing.is_file() and json.loads(existing.read_text()).get('status') == 'complete':
                item.update(status='complete', reused=True); save()
                print(f"REUSED {task['adapter']} complete", flush=True)
                continue
            if task['smoke']:
                child = launch(task['smoke'], directory / 'smoke.log', task.get('smoke_gpus'))
                item.update(status='smoke', pids=[child.pid]); save()
                code = child.wait(); processes.remove(child)
                if code:
                    item.update(status='smoke_failed', exit_code=code); save()
                    print(f"SMOKE FAILED {task['adapter']}; full run skipped; see {directory / 'smoke.log'}", flush=True)
                    continue
            children = []
            if task['mode'] == 'realtime':
                children.append(launch(task['commands'][0], directory / 'supervisor.log'))
            else:
                for rank, worker in enumerate(task['commands']):
                    flags = (['--retry-errors-once'] if args.probe_retry_errors_once else
                             ['--skip-failed'] if args.probe_skip_failed else [])
                    command = worker['argv'] + flags
                    children.append(launch(command, directory / f'worker_{rank}' / 'worker.log', worker['gpus']))
            item.update(status='running', pids=[c.pid for c in children]); save()
            while any(c.poll() is None for c in children):
                if task['mode'] == 'probe':
                    records, errors = [], []
                    for rank in range(len(children)):
                        root = directory / f'worker_{rank}'
                        records += read_jsonl(root / 'probe_predictions.jsonl', complete_only=True)
                        errors += read_jsonl(root / 'probe_errors.jsonl', complete_only=True)
                    atomic_write_text(directory / 'probe_predictions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
                    write_report(load_manifest(Path(task['manifest'])), records, directory, errors)
                    item['completed_probes'] = len(records)
                item['exit_codes'] = [c.poll() for c in children]; save()
                time.sleep(30)
            if task['mode'] == 'probe':
                records, errors = [], []
                for rank in range(len(children)):
                    root = directory / f'worker_{rank}'
                    records += read_jsonl(root / 'probe_predictions.jsonl', complete_only=True)
                    errors += read_jsonl(root / 'probe_errors.jsonl', complete_only=True)
                atomic_write_text(directory / 'probe_predictions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
                atomic_write_text(directory / 'probe_errors.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in errors))
                write_report(load_manifest(Path(task['manifest'])), records, directory, errors)
                item['completed_probes'] = len(records)
            failed_status = ('incomplete_final' if task['mode'] == 'probe' and
                             (args.probe_retry_errors_once or args.probe_skip_failed)
                             else 'incomplete')
            item.update(status='complete' if all(c.returncode == 0 for c in children) else failed_status,
                        exit_codes=[c.returncode for c in children])
            for c in children: processes.remove(c)
            save()
            print(f"END {task['mode']}/{task['adapter']} {item['status']}", flush=True)
        if all(j['status'] == 'complete' for j in status['jobs']):
            status['status'] = 'complete'
        elif all(j['status'] in ('complete', 'incomplete_final') for j in status['jobs']):
            status['status'] = 'complete_with_errors'
        else:
            status['status'] = 'incomplete'
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
    finally:
        terminate_owned(processes)
        for log in logs: log.close()
        save()
        lock.close()
    return int(status['status'] not in ('complete', 'complete_with_errors'))


if __name__ == '__main__':
    raise SystemExit(main())
