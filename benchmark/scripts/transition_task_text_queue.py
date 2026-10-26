#!/usr/bin/env python3
"""Finish frozen Agent jobs, then hand off to a versioned benchmark suite."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.transition.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / 'transition_status.json'
    status = dict(pid=os.getpid(), status='running', stage='current_agent', jobs=[])
    child = None

    def save():
        write(status_path, status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    def launch(command, cwd, label):
        nonlocal child
        with (output / f'{label}.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
                     'HF_HUB_OFFLINE': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})
            status['child_pid'] = child.pid
            save()
            code = child.wait()
            child = None
            return code

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save()
        current = config['current']
        while alive(current['pid']):
            if Path(current['status']).is_file():
                current_status = json.loads(Path(current['status']).read_text())
                status['current_progress'] = {key: current_status.get(key) for key in
                    ('status', 'completed_samples', 'expected_samples')}
                save()
            time.sleep(10)
        current_status = json.loads(Path(current['status']).read_text())
        if current_status['status'] != 'complete':
            raise RuntimeError('Current Agent run is incomplete; refusing automatic handoff')
        for dispatcher in config['old_dispatchers']:
            pid = dispatcher['pid']
            if alive(pid):
                command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
                if dispatcher['script'] not in command:
                    raise RuntimeError('Dispatcher PID identity changed')
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
        # Old dispatchers can only own the now-finished current child. Let them close.
        deadline = time.monotonic() + 60
        while any(alive(d['pid']) for d in config['old_dispatchers']):
            if time.monotonic() > deadline:
                raise RuntimeError('Old dispatchers did not exit; refusing overlapping suites')
            time.sleep(1)
        for job in config['frozen_jobs']:
            status['stage'] = 'frozen_agent'
            item = dict(model=job['model'], status='running')
            status['jobs'].append(item)
            save()
            code = launch(job['argv'], config['frozen_root'], job['model'])
            item.update(status='complete' if code == 0 else 'failed', exit_code=code)
            save()
            if code:
                raise RuntimeError(f"Frozen Agent run failed: {job['model']}")
        status['stage'] = 'awaiting_validated_suite'
        save()
        while not Path(config['ready']).is_file():
            time.sleep(10)
        ready = json.loads(Path(config['ready']).read_text())
        command_path = Path(config['next_argv'][config['next_argv'].index('--commands') + 1])
        if ready['commands_sha256'] != hashlib.sha256(command_path.read_bytes()).hexdigest():
            raise RuntimeError('Validated queue commands changed')
        status['stage'] = 'new_suite'
        save()
        code = launch(config['next_argv'], config['repo'], 'suite')
        status.update(status='complete' if code == 0 else 'incomplete', exit_code=code)
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        save()


if __name__ == '__main__':
    main()
