#!/usr/bin/env python3
"""Run a pinned diagnostic set, then seed its first candidate into the full run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.data.files import write_json
from benchmark.runtime.parallel import terminate_owned
from benchmark.runtime.rollout_stage import prepare, completed_rows, adopt_first, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    children = []
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (root / '.rollouts.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest, commands = prepare(config)
            if args.prepare_only:
                print(json.dumps(dict(status='prepared', qa_count=len(manifest['sample_indices']),
                                      trajectories=len(manifest['sample_indices']) * len(commands))))
                return
            for number, command in enumerate(commands):
                if (root / 'STOP').exists():
                    raise KeyboardInterrupt
                write_json(root / 'status.json', dict(pid=os.getpid(), stage='running', rollout_index=number))
                with (root / f'rollout_{number:02d}.log').open('ab') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                if child.wait():
                    raise RuntimeError(f'Rollout {number} failed; refusing to start the full benchmark')
                children.remove(child)
                completed_rows(root / f'rollout_{number:02d}', manifest['sample_indices'])
            adopt_first(config, manifest)
            summarize(config, manifest)
            write_json(root / 'status.json', dict(pid=os.getpid(), stage='complete'))
        except BaseException as exc:
            write_json(root / 'status.json', dict(pid=os.getpid(), stage='failed', error=str(exc)))
            raise
        finally:
            terminate_owned(children)


if __name__ == '__main__':
    main()
