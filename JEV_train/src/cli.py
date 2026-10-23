import argparse
from pathlib import Path

from .config import ROOT,load


def main():
    parser=argparse.ArgumentParser(description='JEV binary gate and pluggable Omni response backend')
    parser.add_argument('--config',type=Path,default=ROOT/'JEV_train/configs/default.json')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('train-classifier','train-backend','evaluate','stream'):
        p=commands.add_parser(name);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if name in ('evaluate','stream'):
            p.add_argument('--split',choices=('validation','test'),default='test')
            p.add_argument('--sample-index',type=int)
            p.add_argument('--realtime',action='store_true',help='Pace media production at one second per wall second')
    args=parser.parse_args();config=load(args.config)
    from .data import prepare,load_rows
    if args.command=='prepare':prepare(args.data,args.output,config);return
    import hashlib
    config['prepared_manifest_sha256']=hashlib.sha256((args.prepared/'manifest.json').read_bytes()).hexdigest()
    config['prepared_directory']=str(args.prepared.resolve())
    if args.command in ('evaluate','stream'):
        from .pipeline import run
        rows=load_rows(args.prepared,args.split,config)
        if args.sample_index is not None:rows=[r for r in rows if r['sample_index']==args.sample_index]
        if not rows:raise ValueError('No selected prepared rows')
        run(rows,config,args.output,args.realtime);return
    from . import training
    rows=load_rows(args.prepared,'train',config);validation=load_rows(args.prepared,'validation',config)
    getattr(training,'classifier' if args.command=='train-classifier' else 'backend')(rows,validation,config,args.output)
