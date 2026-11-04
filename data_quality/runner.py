"""Resumable eight-replica audit, human-review reports, then the prior AOI queue."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.data.files import write_json,append_jsonl,read_jsonl,atomic_write_text,recover_jsonl_tail
from benchmark.evaluation.timing import reference_windows
from benchmark.data.timestamps import seconds
from data_quality.core import digest,static_findings


def prepare(config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    jobs=[];static=[];sources={};groups=defaultdict(list)
    for dataset in config['datasets']:
        name=dataset['name'];path=Path(dataset['data']);samples=json.loads(path.read_text())
        sources[name]=dict(sha256=digest(path),qa=len(samples),data=str(path),videos_dir=dataset['videos_dir'])
        known={}
        if dataset.get('metadata'):
            for r in json.loads(Path(dataset['metadata']).read_text())['videos']:
                known[r['file']]=known[Path(r['file']).name]=float(r['duration_seconds'])
        prior_exclusions={}
        if dataset.get('audit_report'):
            audit=json.loads(Path(dataset['audit_report']).read_text())
            if audit['source_sha256']!=sources[name]['sha256']:raise ValueError('Prior scheduling audit source mismatch')
            prior_exclusions={r['sample_index']:r for r in audit['excluded']}
        videos=sorted({s['video'] for s in samples})
        def duration(video):
            path=Path(dataset['videos_dir'])/video
            if not path.is_file():return video,None
            if video in known:return video,known[video]
            value=subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
                '-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True,timeout=30)
            return video,float(value.strip())
        with ThreadPoolExecutor(max_workers=config['media_workers']) as pool:durations=dict(pool.map(duration,videos))
        for index,sample in enumerate(samples):
            duration=durations[sample['video']]
            try:
                findings=static_findings(sample,duration) if duration is not None else [dict(code='missing_video',severity='error')]
                refs=reference_windows(sample)
                if not refs:findings.append(dict(code='no_scored_response',severity='error'))
            except (ValueError,KeyError,TypeError) as exc:
                findings=[dict(code='malformed_annotation',severity='error',detail=str(exc))];refs=[]
            if index in prior_exclusions:
                findings.append(dict(code='sft_'+prior_exclusions[index]['reason'],severity='error'))
            static.append(dict(dataset=name,sample_index=index,question=sample.get('question'),video=sample.get('video'),
                               task=sample.get('task'),mode=sample.get('mode'),findings=findings))
            for ri,ref in refs:
                try:cutoff=min(duration,seconds(ref['window_end']))
                except (ValueError,TypeError):continue
                if cutoff<=0:continue
                job=dict(id=f'{name}:{index}:{ri}',dataset=name,sample_index=index,response_index=ri,
                         video=sample['video'],prefix_end_seconds=cutoff)
                groups[(name,sample['video'])].append(job)
    # Keep each video's QA on one worker for bounded in-memory media reuse.
    shards=[[] for _ in config['gpus']];cost=[0.]*len(shards)
    for key,values in sorted(groups.items(),key=lambda p:(-sum(j['prefix_end_seconds'] for j in p[1]),p[0])):
        rank=min(range(len(shards)),key=lambda r:(cost[r],r))
        shards[rank].extend(values);cost[rank]+=sum(j['prefix_end_seconds'] for j in values)
    manifest=dict(version='gemma4_prefix_annotation_audit_v1',sources=sources,model_path=config['model_path'],
                  report_only=True,shards=shards,static=static,settings=config)
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Audit identity changed; use a new output directory')
    write_json(path,manifest)
    print(json.dumps(dict(qa=sum(s['qa'] for s in sources.values()),response_jobs=sum(map(len,shards)),
                         shards=list(map(len,shards)),report_only=True)),flush=True)


def worker(config,rank):
    import gc
    import torch
    from data_quality.gemma import GemmaAudit
    out=Path(config['output']);manifest=json.loads((out/'manifest.json').read_text());jobs=manifest['shards'][rank]
    data={name:json.loads(Path(s['data']).read_text()) for name,s in manifest['sources'].items()}
    for name,source in manifest['sources'].items():
        if digest(source['data'])!=source['sha256']:raise ValueError('Audit source changed')
    path=out/f'worker_{rank}/judgments.jsonl';path.parent.mkdir(exist_ok=True)
    recover_jsonl_tail(path)
    done={r['id'] for r in read_jsonl(path,complete_only=True)}
    pending=[j for j in jobs if j['id'] not in done]
    if not pending:return
    auditor=GemmaAudit(config)
    # Exercise each native replica on its shortest remaining prefix first.
    first=min(pending,key=lambda j:(j['prefix_end_seconds'],j['id']))
    pending=[first]+[j for j in pending if j['id']!=first['id']]
    for job in pending:
        started=time.time()
        try:
            sample=data[job['dataset']][job['sample_index']]
            video=Path(manifest['sources'][job['dataset']]['videos_dir'])/job['video']
            result=auditor.review(sample,job['response_index'],video,job['prefix_end_seconds'])
        except Exception as exc:
            result=dict(status='failed',error_type=type(exc).__name__,error=str(exc))
        append_jsonl(path,dict(job,reference=data[job['dataset']][job['sample_index']]['responses'][job['response_index']],**result,elapsed_seconds=time.time()-started))
        print(json.dumps(dict(id=job['id'],status=result['status'],verdict=result.get('verdict'),seconds=time.time()-started)),flush=True)
        gc.collect();torch.cuda.empty_cache()


def report(config):
    out=Path(config['output']);m=json.loads((out/'manifest.json').read_text());rows=[]
    for rank in range(len(config['gpus'])):rows+=read_jsonl(out/f'worker_{rank}/judgments.jsonl',complete_only=True)
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate audit judgments')
    lines=['# Gemma4 annotation quality audit','',
        '**Report only: no QA, response, timestamp, dataset split or benchmark label was changed.**',
        'Each scored response uses audio/video through its window end only. Video is sparse; uncertain evidence is not a confirmed defect.',
        'Coverage is not a guarantee of annotation correctness. Inspect raw judgments and model failures before making cleaning decisions.','',
        '| Dataset | QA | Static-error QA | Response jobs | Attempted | Pass | Issue | Uncertain | Failed |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    summary={};qa_results=[]
    for name,source in m['sources'].items():
        selected=[r for r in rows if r['dataset']==name];counts=Counter(r.get('verdict') if r['status']=='complete' else 'failed' for r in selected)
        statics=[r for r in m['static'] if r['dataset']==name]
        n=sum(j['dataset']==name for shard in m['shards'] for j in shard)
        errors=sum(any(f['severity']=='error' for f in r['findings']) for r in statics)
        summary[name]=dict(qa=source['qa'],static_error_qa=errors,response_jobs=n,attempted=len(selected),counts=dict(counts),source_sha256=source['sha256'])
        lines.append(f"| {name} | {source['qa']} | {errors} | {n} | {len(selected)} | {counts['pass']} | {counts['issue']} | {counts['uncertain']} | {counts['failed']} |")
        codes=Counter(f['code'] for r in statics for f in r['findings']);checks=Counter(k for r in selected for k,v in r.get('checks',{}).items() if v=='issue')
        summary[name].update(static_findings=dict(codes),model_issue_checks=dict(checks))
        lines.extend(['',f'## {name}', '', 'Static findings: '+json.dumps(dict(codes),ensure_ascii=False),
                      'Gemma issue checks: '+json.dumps(dict(checks),ensure_ascii=False)])
        by_index=defaultdict(list)
        for r in selected:by_index[r['sample_index']].append({k:r.get(k) for k in ['id','response_index','reference','status','verdict','summary','findings','error']})
        for r in statics:qa_results.append(dict(r,response_reviews=by_index[r['sample_index']]))
    write_json(out/'summary.json',dict(updated_at=time.time(),datasets=summary,report_only=True))
    write_json(out/'qa_review.json',qa_results)
    atomic_write_text(out/'report.md','\n'.join(lines)+'\n')
    with (out/'response_review.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['dataset','qa_index','response_index','status','verdict','summary_or_error'])
        for r in rows:writer.writerow([r['dataset'],r['sample_index'],r['response_index'],r['status'],r.get('verdict'),r.get('summary',r.get('error'))])
    return len(rows),sum(len(s) for s in m['shards'])


def run(config,gpus=None):
    import fcntl
    from train.scripts.sft_then_split_bench import previous_ready
    from benchmark.runtime.parallel import terminate_owned
    active_gpus=config['gpus'] if gpus is None else gpus
    if not active_gpus or len(set(active_gpus))!=len(active_gpus) or not set(active_gpus)<=set(config['gpus']):
        raise ValueError('Audit execution GPUs must be a distinct nonempty subset of manifest GPUs')
    out=Path(config['output']);lock=(out/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        # A deferred continuation may arrive after a manual resource reassignment.
        # Keep the active owner and its worker files untouched.
        print('Audit already has an active controller; reusing it.',flush=True)
        return
    children=[]
    def mark(stage,**kw):write_json(out/'status.json',dict(pid=os.getpid(),stage=stage,at=time.time(),**kw))
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        mark('waiting_for_current_benchmarks_and_judge')
        while not previous_ready(config['wait_for_previous']):
            if (out/'STOP').exists():raise KeyboardInterrupt
            time.sleep(config['poll_seconds'])
        deadline=time.time()+config['gpu_release_timeout_seconds']
        while True:
            used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
            if not any(int(line.split(',')[1])>2048 for line in used if line.split(',')[0].strip() in active_gpus):break
            if time.time()>deadline:raise RuntimeError('Audit GPUs still occupied')
            time.sleep(config['poll_seconds'])
        def launch(rank,gpu):
            directory=out/f'worker_{rank}';directory.mkdir(exist_ok=True)
            with (directory/'worker.log').open('ab') as log:
                p=subprocess.Popen([config['python'],'-u','-m','data_quality','--config',config['config_path'],'worker','--rank',str(rank)],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(ROOT),
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p)
            return p
        # Preserve all original shard IDs and JSONL paths when fewer GPUs are
        # available. Each GPU processes one existing shard at a time.
        pending=list(range(len(config['gpus'])));active={};codes=[]
        while pending or active:
            for gpu,(rank,p) in list(active.items()):
                code=p.poll()
                if code is None:continue
                codes.append(code);children.remove(p);del active[gpu]
                if code:raise RuntimeError(f'Audit shard {rank} exited {code}')
            for gpu in active_gpus:
                if gpu not in active and pending:
                    rank=pending.pop(0);active[gpu]=(rank,launch(rank,gpu))
            done,total=report(config)
            mark('gemma4_audit_running',gpus=active_gpus,completed=done,total=total,
                 workers=[dict(pid=p.pid,rank=rank,gpu=gpu) for gpu,(rank,p) in active.items()],pending_shards=pending)
            if (out/'STOP').exists():raise KeyboardInterrupt
            if pending or active:time.sleep(config['poll_seconds'])
        done,total=report(config)
        if any(codes) or done!=total:raise RuntimeError(f'Audit worker failed or coverage incomplete: {done}/{total}, {codes}')
        summary=json.loads((out/'summary.json').read_text())
        successful=sum(sum(v['counts'].get(k,0) for k in ('pass','issue','uncertain')) for v in summary['datasets'].values())
        if successful/max(total,1)<config['minimum_success_fraction']:
            raise RuntimeError('Too many invalid/failed Gemma reviews; report preserved, continuation blocked')
        mark('audit_report_complete',completed=done,total=total,successful=successful)
        # Only report-only completion is automated. Never restart SFT here.
        if config.get('continuation'):
            with (out/'remaining_benchmark.log').open('ab') as log:
                p=subprocess.Popen(config['continuation'],cwd=config['continuation_cwd'],
                    env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p);mark('remaining_aoi_benchmarks',child_pid=p.pid)
            code=p.wait()
            if code:raise RuntimeError(f'Remaining benchmark queue exited {code}')
        mark('complete')
    except KeyboardInterrupt:mark('stopped')
    except BaseException as exc:mark('failed',error=str(exc));raise
    finally:terminate_owned(children)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','report'):sub.add_parser(name)
    p=sub.add_parser('run');p.add_argument('--gpus',nargs='+',help='Execution GPU subset; preserves existing audit shards')
    p=sub.add_parser('worker');p.add_argument('--rank',type=int,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());config['config_path']=str(args.config.resolve())
    if config.get('report_only') is not True:raise ValueError('This entrypoint supports report-only audit, not automatic label changes')
    if args.command=='worker':worker(config,args.rank)
    elif args.command=='run':run(config,args.gpus)
    else:globals()[args.command](config)
