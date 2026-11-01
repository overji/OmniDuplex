#!/usr/bin/env python3
"""Reorder an owned OmniPro queue without changing samples, seeds or shard membership.

Dry run by default. --apply stops the controller and its inference children,
backs up plans, validates regenerated plans, and restarts the same queue.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def shuffled_order(count, workers, seed):
    """Interleaved per-rank permutations preserve existing prediction ownership."""
    shards=[]
    for rank in range(workers):
        indices=list(range(rank,count,workers));random.Random(seed+rank).shuffle(indices);shards.append(indices)
    order=[shards[rank][offset] for offset in range(max(map(len,shards))) for rank in range(workers) if offset<len(shards[rank])]
    return order,shards


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    path=args.config.resolve();cfg=json.loads(path.read_text());out=Path(cfg['output'])
    samples=json.loads(Path(cfg['data']).read_text());widths={len(job['gpus']) for job in cfg['models']}
    if len(widths)!=1:raise ValueError('This resume utility requires equal replica counts')
    order,shards=shuffled_order(len(samples),widths.pop(),args.seed)
    state=json.loads((out/'queue_status.json').read_text())
    if state.get('stage')!='inference_running':raise ValueError('Only live inference can be reordered')
    print(json.dumps(dict(seed=args.seed,samples=len(order),next_tasks=[samples[i]['task'] for i in shards[0][:12]],apply=args.apply),indent=2),flush=True)
    if not args.apply:return
    spec=importlib.util.spec_from_file_location('owned',cfg['queue_helper_script']);watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
    controller=watch.process(state['pid'])
    if not controller or str(path) not in controller['argv'] or 'omnipro' not in controller['argv']:raise RuntimeError('Controller identity differs')
    roots=[]
    for job in state['jobs']:
        entry=watch.process(job['pid'])
        if entry:
            if '-m' not in entry['argv'] or 'parallel' not in entry['argv'] or str(out) not in ' '.join(entry['argv']):raise RuntimeError('Inference supervisor identity differs')
            roots.append(entry)
    # Suspend controller so it cannot treat this planned interruption as failure
    # or restore the prior AOI queue while the GPU handoff is in progress.
    os.kill(controller['pid'],signal.SIGSTOP)
    try:
        watch.stop_owned(roots,cfg['stop_grace_seconds'])
        backup=out/'order_history'/str(int(time.time()));backup.mkdir(parents=True)
        (backup/'queue_config.json').write_text(path.read_text())
        order_path=out/f'sample_order_seed{args.seed}.json';order_path.write_text(json.dumps(order)+'\n')
        for model in cfg['models']:
            directory=out/'runs'/model['label'];plan_path=directory/'parallel_plan.json'
            if plan_path.exists():
                plan=json.loads(plan_path.read_text());(backup/(model['label']+'.parallel_plan.json')).write_text(plan_path.read_text())
                old_shards=plan['assignments']
                if len(old_shards)!=len(shards) or any(set(a)!=set(b) for a,b in zip(old_shards,shards)):
                    raise RuntimeError('Shard membership would change')
                for rank in range(len(shards)):
                    pred=directory/f'worker_{rank}/predictions.jsonl'
                    if pred.exists():
                        data=pred.read_bytes();lines=data[:data.rfind(b'\n')+1].decode().splitlines()
                        if not all(json.loads(line)['sample_index'] in set(shards[rank]) for line in lines):raise RuntimeError('Existing predictions have wrong shard ownership')
                plan['assignments']=shards;plan['sample_indices']=order
                plan_path.write_text(json.dumps(plan,indent=2)+'\n')
            argv=model['argv']
            if '--sample-indices' in argv:argv[argv.index('--sample-indices')+1]=str(order_path)
            else:argv+=['--sample-indices',str(order_path)]
        cfg['execution_order']=dict(kind='random_within_existing_shards',seed=args.seed,sample_indices=str(order_path),data_order_unchanged=True)
        path.write_text(json.dumps(cfg,indent=2)+'\n')
        # Prepare-plan validation uses the frozen code that will run inference.
        validation='''import json,sys\nfrom pathlib import Path\nfrom benchmark.cli.main import build_parser\nfrom benchmark.runtime.parallel import prepare_plan\nc=json.loads(Path(sys.argv[1]).read_text())\nfor m in c['models']:\n a=build_parser().parse_args(m['argv'][3:]);v=vars(a);v.pop('command');model=v.pop('adapter_name');controls={k:v.pop(k) for k in ('gpus','worker_python','poll_seconds','sample_indices')}\n if not (a.output_dir/'parallel_plan.json').exists():continue\n prepare_plan(a,model,controls['gpus'],controls['worker_python'] or c['python'],sample_indices=json.loads(Path(controls['sample_indices']).read_text()))\nprint('All existing plans match frozen runtime')'''
        env={**os.environ,'PYTHONPATH':cfg['runtime_root'],'PYTHONUNBUFFERED':'1'}
        subprocess.run([cfg['python'],'-c',validation,str(path)],cwd=cfg['runtime_root'],env=env,check=True)
        os.kill(controller['pid'],signal.SIGKILL)
        with (out/'controller.log').open('ab') as log:
            child=subprocess.Popen([cfg['python'],'-u','-m','benchmark','omnipro','queue','--config',str(path)],cwd=cfg['runtime_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid,backup=str(backup),reused_completed_results=True)),flush=True)
    except BaseException:
        # Preserve a live controller for diagnosis; do not leave a suspended job.
        current=watch.process(controller['pid'])
        if current:os.kill(controller['pid'],signal.SIGCONT)
        raise

if __name__=='__main__':main()
