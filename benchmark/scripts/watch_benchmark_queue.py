#!/usr/bin/env python3
"""Monitor and resume an owned benchmark queue without changing evaluation."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process(pid):
    if not pid:
        return None
    try:
        root = Path(f'/proc/{int(pid)}')
        stat = (root / 'stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=int(pid), ppid=int(stat[1]), state=stat[0], start=stat[19],
                    argv=(root/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def option(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def marker(directory):
    count = 0
    for name in ('predictions.jsonl', 'probe_predictions.jsonl'):
        path = directory/name
        if path.is_file():
            with path.open('rb') as stream:
                count += sum(bool(line.strip()) and line.endswith(b'\n') for line in stream)
    chunks = [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
              for p in sorted(directory.glob('sample_*.chunks.jsonl'))]
    return count, [count, chunks]


def inspect(config, now):
    output = Path(config['output']); path = output/'transition_status.json'; status = read(path)
    snapshot = dict(status=status, key='transition', completed=0, workers=[])
    def result(verdict, reason):
        return {**snapshot, 'verdict': verdict, 'reason': reason}
    suite_dir = Path(option(config['next_argv'], '--output-dir'))
    suite = read(suite_dir/'suite_status.json')
    if status.get('status') == 'complete' and suite.get('status') in ('complete', 'complete_with_errors'):
        return result('complete', 'all_jobs_complete')
    if status.get('status') == 'interrupted':
        return result('paused', 'queue_intentionally_interrupted')
    directory, parallel, finished = output, {}, set()
    if status.get('stage') == 'current_agent':
        directory = Path(config['current']['status']).parent
        parallel = read(config['current']['status'])
    elif status.get('stage') == 'frozen_agent' and status.get('jobs'):
        job = next(j for j in config['frozen_jobs'] if j['model'] == status['jobs'][-1]['model'])
        directory = Path(option(job['argv'], '--output-dir'))
        parallel = read(directory/'parallel_status.json')
    elif status.get('stage') == 'new_suite' and suite.get('jobs'):
        failed = next((j for j in suite['jobs'] if j['status'] in ('incomplete', 'smoke_failed')), None)
        job = failed or suite['jobs'][-1]; directory = Path(job['output'])
        if job.get('status') == 'smoke':
            task = next(t for t in read(suite_dir/'suite_plan.json')['tasks'] if t['output'] == job['output'])
            directory = Path(option(task['smoke'], '--output-dir'))
        snapshot.update(key=str(directory), completed=marker(directory)[0])
        if failed:
            return result('repair', failed['status'])
        bounded_probe = (job.get('mode') == 'probe' and any(flag in config['next_argv']
                         for flag in ('--probe-retry-errors-once', '--probe-skip-failed')))
        if not bounded_probe and any(code not in (None, 0) for code in job.get('exit_codes', [])):
            return result('repair', 'probe_worker_failed')
        finished = {f'worker_{rank}' for rank, code in enumerate(job.get('exit_codes', []))
                    if code == 0 or (bounded_probe and code is not None)}
        parallel = read(directory/'parallel_status.json')
    snapshot['key'] = str(directory)
    finished.update(f"worker_{w['rank']}" for w in parallel.get('workers', [])
                    if w.get('exit_code') == 0 or (parallel.get('failure_policy') == 'skip'
                                                  and w.get('exit_code') is not None))
    snapshot['completed'] = 0
    for worker in sorted(directory.glob('worker_*')) or [directory]:
        if not worker.is_dir():
            continue
        count, value = marker(worker); snapshot['completed'] += count
        if worker.name not in finished and parallel.get('status') != 'complete':
            snapshot['workers'].append(dict(key=str(worker), marker=value))
    if parallel.get('failure_policy') != 'skip' and any(
            w.get('exit_code') not in (None, 0) for w in parallel.get('workers', [])):
        return result('repair', 'realtime_worker_failed')
    if not process(status.get('pid')) or status.get('status') in ('failed', 'incomplete', 'complete'):
        return result('repair', 'queue_failed_or_exited')
    if status.get('stage') == 'awaiting_validated_suite' and not Path(config['ready']).is_file():
        return result('waiting', 'awaiting_validated_commands')
    recent = now-path.stat().st_mtime < config['startup_grace_seconds']
    if status.get('child_pid') and not process(status['child_pid']) and not recent:
        return result('repair', 'child_exited_without_handoff')
    return result('healthy', 'processes_alive')


def evaluate(snapshot, state, config, now):
    if state.get('pending_repair') and snapshot['verdict'] == 'paused':
        return {**snapshot, 'verdict': 'repair', 'reason': 'continue_own_repair',
                **state['pending_repair']}
    observations = state.setdefault('observations', {})
    if snapshot['verdict'] != 'healthy':
        return snapshot
    for worker in snapshot['workers']:
        previous = observations.get(worker['key'])
        if previous is None or previous['marker'] != worker['marker']:
            observations[worker['key']] = dict(marker=deepcopy(worker['marker']), changed_at=now)
        elif now-previous['changed_at'] >= config['stale_after_seconds']:
            return {**snapshot, 'verdict': 'repair', 'reason': 'worker_no_progress',
                    'stalled_worker': worker['key']}
    return snapshot


def owned_roots(config, status, transition_path):
    roots = []
    parent = process(status.get('pid'))
    if parent:
        if (str(ROOT/'benchmark/scripts/transition_task_text_queue.py') not in parent['argv']
                or option(parent['argv'], '--config') != str(transition_path)):
            raise RuntimeError('Queue PID identity differs; refusing to signal it')
        roots.append(parent)
    child = process(status.get('child_pid'))
    if child:
        commands = [j['argv'] for j in config['frozen_jobs']] + [config['next_argv']]
        if not any(child['argv'][1:] == argv[1:] for argv in commands):
            raise RuntimeError('Child PID identity differs; refusing to signal it')
        roots.append(child)
    return roots


def stop_owned(roots, grace):
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            entry = process(int(path.name))
            if entry:
                processes[entry['pid']] = entry
    owned = {p['pid']:p for p in roots if p['pid'] in processes
             and processes[p['pid']]['start'] == p['start']}
    roots = [p for p in roots if p['pid'] in owned]
    while True:
        extra = {pid:p for pid,p in processes.items() if p['ppid'] in owned and pid not in owned}
        if not extra:
            break
        owned.update(extra)
    def signal_same(entry, sig):
        current = process(entry['pid'])
        if current and current['start'] == entry['start']:
            try:
                os.kill(entry['pid'], sig)
            except ProcessLookupError:
                pass
    for entry in roots:
        signal_same(entry, signal.SIGTERM)
        if entry['state'] in ('T', 't'):
            signal_same(entry, signal.SIGCONT)
    deadline = time.monotonic()+grace
    while any(process(pid) for pid in owned) and time.monotonic() < deadline:
        time.sleep(1)
    for entry in owned.values():
        signal_same(entry, signal.SIGKILL)
    return sorted(owned)


def occupied_gpus(gpus):
    def query(value):
        return subprocess.run(['nvidia-smi', value, '--format=csv,noheader,nounits'],
                              check=True, capture_output=True, text=True, timeout=30).stdout.splitlines()
    selected = {line.split(',')[1].strip() for line in query('--query-gpu=index,uuid')
                if line.split(',')[0].strip() in gpus}
    return [line for line in query('--query-compute-apps=pid,gpu_uuid')
            if ',' in line and line.split(',')[1].strip() in selected]


def repair(snapshot, state, config, transition_path, output, now):
    retry = state.setdefault('retries', {}).setdefault(snapshot['key'], dict(completed=-1, attempts=0))
    if snapshot['completed'] > retry['completed']:
        retry.update(completed=snapshot['completed'], attempts=0)
    if retry['attempts'] >= config['max_restarts_without_progress']:
        return dict(action='needs_attention', reason='retry_budget_exhausted', key=snapshot['key'])
    state['pending_repair'] = {k:snapshot[k] for k in ('key', 'completed')}
    roots = owned_roots(config, snapshot['status'], transition_path)
    stopped = stop_owned(roots, config['stop_grace_seconds'])
    gpus = set(config['next_argv'][config['next_argv'].index('--gpus')+1:])
    busy = occupied_gpus(gpus)
    if busy:
        return dict(action='waiting_for_gpus', occupied=busy, stopped=stopped)
    write(output/f'transition_before_restart_{int(now)}.json', snapshot['status'])
    with (output/'restarts.log').open('ab') as log:
        child = subprocess.Popen([sys.executable,'-u',str(ROOT/'benchmark/scripts/transition_task_text_queue.py'),
            '--config',str(transition_path)], cwd=config['repo'], stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    retry['attempts'] += 1
    state.pop('pending_repair', None)
    state['observations'] = {}
    return dict(action='restarted', restarted_pid=child.pid, attempt=retry['attempts'], stopped=stopped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transition-config', type=Path, required=True)
    parser.add_argument('--settings', type=Path, default=ROOT/'benchmark/configs/watchdog.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Inspect only; never stop or launch jobs')
    args = parser.parse_args(); transition_path = args.transition_config.resolve()
    config = {**read(transition_path), **read(args.settings)}
    for key in ('interval_seconds','stale_after_seconds','startup_grace_seconds','stop_grace_seconds','max_restarts_without_progress'):
        if type(config.get(key)) is not int or config[key] <= 0:
            parser.error(f'{key} must be a positive integer')
    output = (args.output_dir or Path(config['output'])/'watchdog').resolve(); output.mkdir(parents=True,exist_ok=True)
    lock = (output/'.watchdog.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = read(output/'state.json')
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        while True:
            now = time.time()
            try:
                snapshot = (dict(verdict='paused',reason='PAUSE_file') if (output/'PAUSE').exists()
                            else evaluate(inspect(config,now),state,config,now))
                action = dict(action='observe')
                if snapshot['verdict']=='repair' and not args.dry_run:
                    action = repair(snapshot,state,config,transition_path,output,now)
                record = dict(pid=os.getpid(),checked_at=now,next_check_at=now+config['interval_seconds'],health=snapshot,**action)
            except Exception as error:
                record = dict(pid=os.getpid(),checked_at=now,next_check_at=now+config['interval_seconds'],
                              action='needs_attention',error=f'{type(error).__name__}: {error}')
            write(output/'status.json',record)
            if not args.dry_run:
                write(output/'state.json',state)
            with (output/'checks.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if args.once or record.get('health',{}).get('verdict')=='complete':
                break
            time.sleep(config['interval_seconds'])
    except KeyboardInterrupt:
        write(output/'status.json',dict(pid=os.getpid(),action='stopped',stopped_at=time.time(),
                                      reason='watchdog_stopped_queue_left_untouched'))


if __name__=='__main__':
    main()
