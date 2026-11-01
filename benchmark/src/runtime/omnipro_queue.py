"""Owned, restartable OmniPro insertion; inference pairs and four judge replicas.

Only this controller may manipulate its original queue. Existing predictions
and frozen configs are reused; training processes are never signalled.
"""
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from benchmark.data.files import write_json
from benchmark.evaluation.omnipro import report, score_run


def helper(path):
    spec=importlib.util.spec_from_file_location('omnipro_owned_queue_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def final_ready(job, process):
    """A directory alone cannot prove a distributed final save completed."""
    output=Path(job['training_output']);checkpoint=output/'checkpoint-final'
    if process(job.get('training_pid')):
        return False
    reports=[output/f'memory_report.rank{rank}.json' for rank in range(job['training_world_size'])]
    if not all(p.is_file() for p in reports):return False
    if not (checkpoint/'training_config.json').is_file():return False
    if not all((checkpoint/f'rng_rank{rank}.pt').is_file() for rank in range(job['training_world_size'])):return False
    if len(list((checkpoint/'state').glob('*optim_states.pt')))!=job['training_world_size']:return False
    for p in reports:
        r=json.loads(p.read_text())
        if not r.get('checkpoint_saved') or r.get('optimizer_steps',0)<job['expected_steps'] or not math.isfinite(r['last_loss']):return False
    return True


class Queue:
    def __init__(self, path):
        self.path=Path(path).resolve();self.config=json.loads(self.path.read_text());self.root=Path(self.config['output'])
        self.root.mkdir(parents=True,exist_ok=True)
        self.watch=helper(self.config['queue_helper_script']);self.child=[]
        self.env={**os.environ,'PYTHONPATH':self.config['runtime_root'],'PYTHONUNBUFFERED':'1',
                  'CUDA_DEVICE_ORDER':'PCI_BUS_ID','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        self.paused=False

    def state(self, stage, **kw):
        write_json(self.root/'queue_status.json',dict(pid=os.getpid(),stage=stage,at=time.time(),**kw))

    def launch(self, argv, log, env=None, cwd=None):
        with Path(log).open('ab') as stream:
            p=subprocess.Popen(argv,cwd=cwd or self.config['runtime_root'],env=env or self.env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.child.append(p)
        return p

    def stop_children(self):
        for p in self.child:
            if p.poll() is None:
                entry=self.watch.process(p.pid)
                if entry:self.watch.stop_owned([entry],self.config['stop_grace_seconds'])
        self.child=[]

    def pause_previous(self):
        previous=Path(self.config['previous_transition']);cfg=json.loads(previous.read_text());directory=Path(cfg['output'])
        pause=directory/'watchdog/PAUSE'
        if pause.exists() and pause.read_text().strip()!=str(self.path):
            raise RuntimeError('Previous queue is paused by another owner')
        pause.write_text(str(self.path)+'\n')
        self.paused=True
        # Snapshot/verify all roots before signalling: a watcher can own the
        # transition as a descendant, and stopping it already tears down children.
        roots=[]
        status=self.watch.read(directory/'watchdog/status.json')
        entry=self.watch.process(status.get('pid'))
        if entry:
            if str(previous) not in entry['argv'] or 'watch_benchmark_queue.py' not in ' '.join(entry['argv']):raise RuntimeError('Watchdog PID identity differs')
            roots.append(entry)
        state=self.watch.read(directory/'transition_status.json')
        parent=self.watch.process(state.get('pid'))
        if parent:
            if str(previous) not in parent['argv'] or 'transition_task_text_queue.py' not in ' '.join(parent['argv']):raise RuntimeError('Queue PID identity differs')
            roots.append(parent)
        child=self.watch.process(state.get('child_pid'))
        if child:
            if str(previous.parent/'commands.json') not in child['argv']:raise RuntimeError('Suite PID identity differs')
            roots.append(child)
        self.watch.stop_owned(roots,self.config['stop_grace_seconds'])
        self.paused=True
        self.wait_free(self.config['gpus'])
        self.state('previous_queue_paused')

    def wait_free(self,gpus):
        deadline=time.monotonic()+self.config['release_timeout_seconds']
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
            busy=[line for line in lines if line.split(',')[0].strip() in gpus and int(line.split(',')[1])>self.config['free_memory_threshold_mib']]
            if not busy:return
            if time.monotonic()>deadline:raise RuntimeError('Owned GPUs remain occupied; refusing overlapping jobs')
            time.sleep(2)

    def resume_previous(self):
        if not self.paused:return
        previous=Path(self.config['previous_transition']);cfg=json.loads(previous.read_text());directory=Path(cfg['output'])
        pause=directory/'watchdog/PAUSE'
        if pause.exists() and pause.read_text().strip()==str(self.path):pause.unlink()
        # Prior runtime and environment are preserved; do not inject OmniPro imports.
        env=dict(os.environ,PYTHONUNBUFFERED='1');env.pop('PYTHONPATH',None)
        status=self.watch.read(directory/'transition_status.json')
        if not self.watch.process(status.get('pid')):
            argv=[self.config['python'],'-u',self.config['transition_script'],'--config',str(previous)]
            self.launch(argv,directory/'transition.log',env,cfg['repo'])
        watcher=self.watch.read(directory/'watchdog/status.json')
        if not self.watch.process(watcher.get('pid')):
            self.launch([self.config['python'],'-u',self.config['queue_helper_script'],'--transition-config',str(previous)],directory/'watchdog.log',env,self.config['repo'])
        # Resumed jobs belong to the original queue, not this controller's cleanup.
        self.child=[];self.paused=False

    def monitor(self, jobs):
        """15-minute health checks; bounded retries preserve existing outputs."""
        pending=[]
        for argv,log in jobs:
            p=self.launch(argv,log);pending.append(dict(process=p,argv=argv,log=log,restarts=0))
        last_health=0
        while True:
            for item in pending:
                p=item['process']
                if p.poll() not in (None,0) and item['restarts']<self.config['max_restarts']:
                    # Changed config/plan errors must fail, not loop forever.
                    item['process']=self.launch(item['argv'],item['log']);item['restarts']+=1
            if all(item['process'].poll() is not None for item in pending):break
            now=time.time()
            for item in pending:
                if item['process'].poll() is not None:continue
                argv=item['argv'];directory=Path(argv[argv.index('--output-dir')+1])
                files=list(directory.glob('worker_*/predictions.jsonl'))+list(directory.glob('worker_*/sample_*.chunks.jsonl'))
                marker=max((p.stat().st_mtime for p in files),default=0)
                if marker!=item.get('marker'):
                    item.update(marker=marker,changed_at=now)
                elif now-item['changed_at']>self.config['stale_after_seconds']:
                    entry=self.watch.process(item['process'].pid)
                    if entry:self.watch.stop_owned([entry],self.config['stop_grace_seconds'])
                    if item['restarts']>=self.config['max_restarts']:raise RuntimeError('Inference stale after bounded retries')
                    item['process']=self.launch(item['argv'],item['log']);item['restarts']+=1;item['changed_at']=now
            if now-last_health>=self.config['health_interval_seconds']:
                self.state('inference_running',jobs=[dict(pid=x['process'].pid,returncode=x['process'].poll(),restarts=x['restarts']) for x in pending])
                last_health=now
            time.sleep(self.config['poll_seconds'])
        codes=[x['process'].returncode for x in pending]
        self.child=[]
        if any(code!=0 for code in codes):raise RuntimeError(f'Inference failed after bounded restart: {codes}')

    def run_models(self,jobs):
        # Require a minimal real native test before full-data inference.
        smoke=[]
        for job in jobs:
            if (self.root/'runs'/job['label']/'omnipro_complete.json').exists():continue
            argv=list(job['argv']);argv[argv.index('--output-dir')+1]=str(self.root/'smoke'/job['label'])
            argv+=['--sample-indices',self.config['smoke_indices']]
            smoke.append((argv,self.root/(job['label']+'.smoke.log')))
        if smoke:
            self.state('native_smoke');self.monitor(smoke)
            for argv,_ in smoke:
                directory=Path(argv[argv.index('--output-dir')+1])
                state=json.loads((directory/'parallel_status.json').read_text())
                if state.get('partial_sample_indices') or state.get('missing_sample_indices'):
                    raise RuntimeError('Native smoke contains failed or missing samples')
        full=[(job['argv'],self.root/(job['label']+'.log')) for job in jobs
              if not (self.root/'runs'/job['label']/'omnipro_complete.json').exists()]
        if full:self.monitor(full)
        for job in jobs:
            directory=self.root/'runs'/job['label'];s=json.loads((directory/'parallel_status.json').read_text())
            if s['missing_sample_indices']:raise RuntimeError('Inference has missing QA')
            # Terminal error rows are preserved and reported, not silently retried forever.
            write_json(directory/'omnipro_complete.json',dict(complete=True,partial_indices=s['partial_sample_indices']))
        self.wait_free(self.config['gpus'])
        self.state('judge_starting',gpus=self.config['gpus'])
        servers=[]
        try:
            for path in self.config['judge_service_configs']:
                servers.append(self.launch([self.config['python'],self.config['judge_service_script'],'--config',path,'--start'],self.root/(Path(path).stem+'.log')))
            deadline=time.monotonic()+self.config['judge_startup_timeout_seconds']
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while True:
                if any(p.poll() is not None for p in servers):raise RuntimeError('Judge service exited during startup')
                ready=True
                for url in self.config['judge']['base_urls']:
                    try:
                        with opener.open(url+'/models',timeout=3) as response:models=json.load(response)
                        ready &= self.config['judge']['model'] in {r['id'] for r in models['data']}
                    except Exception:ready=False
                if ready:break
                if time.monotonic()>deadline:raise RuntimeError('Judge service startup timeout')
                time.sleep(self.config['poll_seconds'])
            for job in jobs:
                self.state('judging',model=job['label'],replicas=len(servers))
                result=score_run(self.config,job['label'])
                write_json(self.root/'judge'/job['label']/'stage.json',dict(status=result))
                report(self.path)
        finally:self.stop_children()
        self.wait_free(self.config['gpus']);report(self.path)

    def final_job(self, job):
        model=Path(job['model_path'])
        if not (model/'full_training_model.json').is_file():
            self.state('exporting_final_checkpoint')
            argv=[self.config['train_python'],'-m','train','--config',job['training_config'],'export',
                  '--checkpoint',str(Path(job['training_output'])/'checkpoint-final'),'--output',str(model)]
            p=self.launch(argv,self.root/'export_final.log');code=p.wait();self.child=[]
            if code:raise RuntimeError('Final model export failed; preserve partial export for inspection')
        self.run_models([job])

    def run(self):
        with (self.root/'.omnipro.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            initial=self.config['models'][:-1];final=self.config['models'][-1]
            try:
                self.pause_previous();self.run_models(initial);self.resume_previous()
                self.state('waiting_for_final_sft',training_output=final['training_output'])
                while not final_ready(final,self.watch.process):
                    if not self.watch.process(final['training_pid']):
                        self.state('waiting_for_training_completion_markers',training_output=final['training_output'])
                    time.sleep(self.config['poll_seconds'])
                self.pause_previous();self.final_job(final);self.resume_previous();self.state('complete')
            except BaseException as exc:
                self.stop_children()
                self.state('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
                if self.paused:self.resume_previous()
                raise


def run(path):
    queue=Queue(path)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    queue.run()
