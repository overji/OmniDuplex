#!/usr/bin/env python3
"""Resume the remaining full suite after an owned Agent recheck releases its GPUs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.data.files import read_jsonl, write_json
from benchmark.runtime.parallel import terminate_owned
from benchmark.scripts.run_prepared_suite import option


def remaining_indices(samples, videos, records):
    completed = {r['sample_index'] for r in records if r.get('status') == 'complete'}
    return [i for i, s in enumerate(samples)
            if s.get('mode') != 'agent' and (videos / s['video']).is_file() and i not in completed]


def process_identity(pid):
    try:
        # starttime guards against a later process reusing this PID.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-for', type=Path, required=True)
    p.add_argument('--commands', type=Path, required=True)
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root = a.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    agent = a.wait_for.resolve()
    recheck = json.loads((agent / 'recheck_plan.json').read_text())
    if a.gpus != recheck['gpus']:
        raise ValueError('Continuation GPUs must match the Agent recheck')
    entries = json.loads(a.commands.read_text())
    base = entries[0]['argv']
    data = Path(base[base.index('--data') + 1])
    videos = Path(base[base.index('--videos-dir') + 1])
    if hashlib.sha256(data.read_bytes()).hexdigest() != recheck['data_sha256']:
        raise ValueError('Dataset changed since Agent recheck')
    samples = json.loads(data.read_text())
    previous = a.previous_root.resolve()
    old_gander = previous / 'realtime/gander/predictions.jsonl'
    pending = remaining_indices(samples, videos, read_jsonl(old_gander, complete_only=True))
    indices_path = root / 'gander_remaining_nonagent.json'
    write_json(indices_path, pending)
    commands = []
    for entry in entries:
        if entry['mode'] == 'realtime' and entry['adapter'] == 'minicpmo':
            continue
        if entry['mode'] == 'realtime' and entry['adapter'] == 'gander':
            if not pending:
                continue
            entry = {**entry, 'argv': option(entry['argv'], '--sample-indices', indices_path)}
        if entry['mode'] == 'realtime':
            entry = {**entry, 'argv': option(entry['argv'], '--agent-call-schema', recheck.get('agent_call_schema', 'legacy'))}
        commands.append(entry)
    command_path = root / 'remaining_commands.json'
    write_json(command_path, commands)
    argv = [sys.executable, str(REPO / 'benchmark/scripts/run_prepared_suite.py'),
            '--commands', str(command_path), '--output-dir', str(root / 'suite'), '--gpus', *a.gpus]
    plan = {'wait_for': str(agent), 'gpus': a.gpus, 'gander_remaining_nonagent': pending,
            'commands': commands, 'argv': argv,
            'agent_plan_sha256': hashlib.sha256((agent / 'recheck_plan.json').read_bytes()).hexdigest(),
            'reused_realtime_sources': {
                name: {'nonagent': str(previous / 'realtime' / name), 'agent': str(agent / name)}
                for name in ('minicpmo', 'gander')},
            'note': 'Use only non-Agent rows from previous results; Agent rows come from the recheck.'}
    plan_path = root / 'continuation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Continuation plan changed; use a new output directory')
    write_json(plan_path, plan)
    if a.dry_run:
        subprocess.run([*argv, '--dry-run'], cwd=REPO, check=True)
        print(f'Waiting plan prepared: {len(pending)} Gander non-Agent QA, {len(commands)} jobs; no inference started')
        return
    state = {'pid': os.getpid(), 'status': 'waiting_for_agent', 'agent_root': str(agent),
             'suite_root': str(root / 'suite'), 'gpus': a.gpus}
    children = []
    def save():
        write_json(root / 'continuation_status.json', state)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        initial = json.loads((agent / 'recheck_status.json').read_text())
        pid = initial['pid']; identity = process_identity(pid)
        state['wait_pid'] = pid; save()
        while True:
            current = json.loads((agent / 'recheck_status.json').read_text())
            if current['pid'] != pid:
                raise RuntimeError('Agent supervisor changed; refusing overlapping GPU jobs')
            alive = identity is not None and process_identity(pid) == identity
            if not alive:
                if current['status'] not in ('complete', 'incomplete'):
                    raise RuntimeError(f"Agent supervisor stopped with status {current['status']}")
                if {j['model'] for j in current['jobs']} != {'minicpmo', 'gander'}:
                    raise RuntimeError('Both Agent jobs must finish before continuation')
                state['agent_status'] = current['status']
                break
            save(); time.sleep(30)
        if hashlib.sha256((agent / 'recheck_plan.json').read_bytes()).hexdigest() != plan['agent_plan_sha256']:
            raise RuntimeError('Agent plan changed while waiting')
        for name, digest in recheck['protocol_hashes'].items():
            if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Protocol changed while waiting: {name}')
        with (root / 'suite.log').open('ab') as log:
            child = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append(child)
            state.update(status='running', child_pid=child.pid); save()
            code = child.wait(); children.remove(child)
        suite = json.loads((root / 'suite/suite_status.json').read_text())
        state.update(status=suite['status'], exit_code=code)
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        terminate_owned(children); save()


if __name__ == '__main__':
    main()
