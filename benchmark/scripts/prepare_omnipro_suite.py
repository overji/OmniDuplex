#!/usr/bin/env python3
"""Prepare an immutable OmniPro inference/Judge insertion, without starting jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.data.files import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--previous-transition',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--old-sft-model',type=Path,required=True)
    p.add_argument('--training-output',type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--training-config',type=Path,required=True)
    p.add_argument('--final-export',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/'queue_config.json'
    if target.exists():raise ValueError('Suite already prepared; use existing config to resume')
    runtime=out/'runtime';runtime.mkdir()
    for package in ('benchmark','train'):
        dest=runtime/package;dest.mkdir()
        for name in ('src','configs','scripts'):
            src=ROOT/package/name
            if src.exists():shutil.copytree(src,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('__init__.py','__main__.py'):
            src=ROOT/package/name
            if src.exists():shutil.copy2(src,dest/name)
    # Prompt tool definitions are shared by the existing benchmark package.
    (runtime/'data_generation').mkdir()
    shutil.copytree(ROOT/'data_generation/src',runtime/'data_generation/src',ignore=shutil.ignore_patterns('__pycache__'))
    hashes={str(f.relative_to(runtime)):hashlib.sha256(f.read_bytes()).hexdigest() for f in runtime.rglob('*') if f.is_file()}
    write_json(out/'runtime_manifest.json',hashes)
    data=out/'inputs/all_qa.json';samples=json.loads(data.read_text());meta=json.loads((out/'inputs/manifest.json').read_text())
    bad={r['sample_index'] for r in meta['anomalies'] if r['window_unreachable']}
    short=min((i for i,s in enumerate(samples) if i not in bad and s['duration_seconds']>=30),key=lambda i:samples[i]['duration_seconds'])
    multi=min((i for i,s in enumerate(samples) if i not in bad and i!=short and len(s['responses'])>1 and s['duration_seconds']>=60),key=lambda i:samples[i]['duration_seconds'])
    long=min((i for i,s in enumerate(samples) if i not in bad and s['duration_seconds']>=300),key=lambda i:samples[i]['duration_seconds'])
    write_json(out/'smoke_indices.json',[short,multi,long])
    python=str(ROOT/'benchmark/.venv/bin/python');train_python=str(ROOT/'train/.venv/bin/python')
    service=json.loads((ROOT/'train/configs/judge_qwen27b_service.json').read_text())
    service['python']=str((ROOT/service['python']).absolute())
    services=[];urls=[]
    for gpu,port in zip((4,5,6,7),(8914,8915,8916,8917)):
        cfg=dict(service,gpu=gpu,port=port);path=out/f'judge_gpu{gpu}.json';write_json(path,cfg);services.append(str(path));urls.append(f'http://127.0.0.1:{port}/v1')
    judge=json.loads((ROOT/'train/configs/minicpmo_full_grpo_confusion_judge.json').read_text())['judge']
    judge.update(base_url=urls[0],base_urls=urls,concurrency=32,system_prompt=str(runtime/'benchmark/configs/prompts/judge.txt'),cache_dir=str(out/'judge_cache'))
    jobs=[]
    for label,model,gpus in [('base',args.base_model,['4','5']),('sft_old500',args.old_sft_model,['6','7']),('sft_fullvideo_final',args.final_export,['6','7'])]:
        argv=[python,'-m','benchmark','parallel','minicpmo','--data',str(data),'--videos-dir',str(args.dataset.resolve()),
            '--model-path',str(model.resolve()),'--output-dir',str(out/'runs'/label),
            '--system-prompt',str(runtime/'benchmark/configs/prompts/system_prompt.txt'),
            '--cache-dir',str(out/'media_cache'),'--agent-call-schema','task-start',
            '--agent-output-protocol','task-text','--decode-mode','sampling','--seed','42',
            '--max-seconds','0','--max-speak-tokens','20','--sliding-window-mode','off',
            '--unused-tts-device','cpu','--cpu-threads','4','--gpus',*gpus]
        jobs.append(dict(label=label,model_path=str(model.resolve()),gpus=gpus,argv=argv))
    training=json.loads(args.training_config.read_text());prepared=ROOT/'train/prepared/full_video_event_65536_20261029/train.json'
    import math
    steps=math.ceil(len(json.loads(prepared.read_text()))/(4*training['gradient_accumulation_steps']))*training['epochs']
    jobs[-1].update(training_output=str(args.training_output.resolve()),training_pid=args.training_pid,
                    training_world_size=4,training_config=str(args.training_config.resolve()),expected_steps=steps)
    cfg=dict(output=str(out),repo=str(ROOT),runtime_root=str(runtime),data=str(data),python=python,train_python=train_python,
        models=jobs,gpus=['4','5','6','7'],previous_transition=str(args.previous_transition.resolve()),
        queue_helper_script=str(ROOT/'benchmark/scripts/watch_benchmark_queue.py'),
        transition_script=str(ROOT/'benchmark/scripts/transition_task_text_queue.py'),
        smoke_indices=str(out/'smoke_indices.json'),judge=judge,judge_service_configs=services,
        judge_service_script=str(runtime/'train/scripts/serve_grpo_judge.py'),
        poll_seconds=10,health_interval_seconds=900,stale_after_seconds=3600,max_restarts=2,
        stop_grace_seconds=20,release_timeout_seconds=120,free_memory_threshold_mib=2048,
        judge_startup_timeout_seconds=1200)
    # Validate old export integrity before pausing any live queue.
    from train.src.infrastructure.checkpoints import full_model_identity
    full_model_identity(args.old_sft_model)
    write_json(target,cfg)
    print(json.dumps(dict(config=str(target),smoke=[dict(index=i,seconds=samples[i]['duration_seconds']) for i in (short,multi,long)],models=[j['label'] for j in jobs]),indent=2))

if __name__=='__main__':main()
