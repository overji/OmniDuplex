#!/usr/bin/env python3
"""Backfill/follow rank-zero JSONL metrics without restarting a training process."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.data.files import write_json
from train.src.infrastructure.tensorboard import create_sink
from train.src.infrastructure.tracking import Tracking


def ingest(record, tracker, step):
    if 'validation_loss' in record:
        components={k[len('validation_'):]:v for k,v in record.items()
                    if k.startswith('validation_') and (k.startswith('validation_loss_') or k.endswith(('_tokens','_qa')))}
        tracker.log_validation(step,record['validation_loss'],record.get('validation_scored',0),
                               record.get('validation_excluded_count',0),components)
    elif 'step' in record:
        step=record['step'];tracker.log_train(record)
    return step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--log-dir',type=Path,default=ROOT/'train/tensorboard')
    p.add_argument('--follow',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=5)
    args=p.parse_args()
    if args.poll_seconds<=0:p.error('poll-seconds must be positive')
    output=args.log_dir/args.run_dir.name;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=args.run_dir/'metrics.rank0.jsonl';identity=source.stat()
    state_path=output/'bridge_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(offset=0,step=0,pending=[])
    signature=[str(source.resolve()),identity.st_dev,identity.st_ino]
    if state.get('source',signature)!=signature or source.stat().st_size<state['offset']:
        raise ValueError('Metrics were replaced or truncated; use a new log directory')
    sink,_=create_sink(dict(log_dir=str(args.log_dir.resolve()),flush_seconds=5),args.run_dir)
    sink.pending=state.get('pending',[]);tracker=Tracking(tensorboard=sink)
    running=True
    def stop(signum,frame):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with source.open('rb') as stream:
            stream.seek(state['offset'])
            while running:
                while True:
                    offset=stream.tell();line=stream.readline()
                    if not line or not line.endswith(b'\n'):
                        stream.seek(offset);break
                    record=json.loads(line)
                    state['step']=ingest(record,tracker,state['step'])
                    state['offset']=stream.tell()
                sink.writer.flush()
                state.update(source=signature,pending=sink.pending,updated_at=time.time())
                write_json(state_path,state)
                if not args.follow or (output/'STOP').exists():break
                time.sleep(args.poll_seconds)
    finally:
        sink.close()


if __name__=='__main__':main()
