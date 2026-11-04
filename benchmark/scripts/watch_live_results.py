#!/usr/bin/env python3
"""Refresh paired Markdown summaries without loading models or controlling GPUs."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from benchmark.evaluation.live_report import render,comparison_section
from benchmark.data.files import atomic_write_text,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    c=json.loads(args.config.read_text());output=Path(c['output']);output.parent.mkdir(parents=True,exist_ok=True)
    if c['interval_seconds']<1:raise ValueError('interval_seconds must be positive')
    lock=output.with_suffix('.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        if args.once:
            # A live watcher already owns this report. A redundant final refresh
            # must not turn completed inference/Judge into a failed GPU queue.
            print(f'Report already owned by an active watcher: {output}',flush=True)
            return
        raise
    running=True
    def stop(signum,frame):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while running:
        lines=['# Latest SFT benchmark — live results','',f'Updated: {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}. Refresh every {c["interval_seconds"]} seconds.','']
        statuses=[]
        for report in c['reports']:
            try:
                pairs=render({**report,'interval_seconds':c['interval_seconds']});status=dict(title=report['title'],status='updated',paired_qa=pairs)
                data=json.loads(Path(report['output']).with_suffix('.json').read_text())
                lines.extend(comparison_section(report,data))
            except Exception as exc:
                status=dict(title=report['title'],status='error',error_type=type(exc).__name__,error=str(exc));lines.append(f"- {report['title']}: update failed ({type(exc).__name__}); inspect watcher status.")
            statuses.append(status)
        atomic_write_text(output,'\n'.join(lines)+'\n');write_json(output.with_suffix('.status.json'),dict(updated_at=time.time(),reports=statuses))
        if args.once or output.with_suffix('.STOP').exists():break
        time.sleep(c['interval_seconds'])


if __name__=='__main__':main()
