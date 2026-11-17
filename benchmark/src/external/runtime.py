"""Explicit foreground execution only. Never edit or pause another experiment queue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import time

from benchmark.data.files import write_json
from benchmark.data.interactscope import digest, local_asset
from benchmark.external.preparation import implementation_identity
from benchmark.external.reporting import report

REPO = Path(__file__).resolve().parents[3]


def validate(config):
    if config['implementation_sha256'] != implementation_identity():
        raise ValueError('External protocol implementation changed; prepare a fresh experiment')
    if digest(config['data']) != config['data_sha256']:
        raise ValueError('Prepared dataset changed')
    meta = json.loads((Path(config['data']).parent/'manifest.json').read_text())
    if meta['source_sha256'] != config['source_sha256']:
        raise ValueError('Source revision changed')
    media_path = Path(config['data']).parent/'media_manifest.json'
    if digest(media_path)!=config['media_manifest_sha256']:
        raise ValueError('Prepared media manifest changed')
    for row in json.loads(media_path.read_text()):
        stat = local_asset(config['dataset_root'],row['video']).stat()
        if stat.st_size!=row['size_bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('Prepared video changed; use a new media cache and experiment directory')
    labels = [m['label'] for m in config['models']]
    if len(labels)!=len(set(labels)):
        raise ValueError('Duplicate model labels')
    for model in config['models']:
        from train.scripts.benchmark_after_training import external_weights
        external_weights(model['model_path'])


def training_ready(job):
    from train.scripts.benchmark_after_training import completed_training
    return completed_training(dict(training_output=job['training_output'],
        training_run_sha256=job['training_run_sha256'], expected_optimizer_steps=job['expected_optimizer_steps']))


def export_ready(job):
    model = Path(job['model_path']).resolve()
    marker = Path(job['export_ready_marker'])
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    metadata = model/'full_training_model.json'
    if (Path(record['model']).resolve()!=model or not metadata.is_file()
            or digest(metadata)!=record['metadata_sha256']
            or Path(json.loads(metadata.read_text())['source_checkpoint']).resolve()
               != (Path(job['training_output'])/'checkpoint-final').resolve()):
        raise ValueError('Final SFT export identity mismatch')
    return True


def ensure_export(config, job):
    checkpoint = training_ready(job)
    if export_ready(job):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    subprocess.run([config['train_python'], '-m', 'train', 'export', '--checkpoint', str(checkpoint),
                    '--output', job['model_path']], cwd=REPO, env=env, check=True)
    metadata = Path(job['model_path'])/'full_training_model.json'
    if Path(json.loads(metadata.read_text())['source_checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Export did not use the selected final checkpoint')
    marker = Path(job['export_ready_marker'])
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker,dict(model=job['model_path'], metadata_sha256=digest(metadata)))
    if not export_ready(job):
        raise ValueError('Export marker missing')


def command(config, job, gpus):
    args = [config['python'], '-u', '-m', 'benchmark', 'parallel', 'minicpmo',
            '--model-path', job['model_path'], '--data', config['data'], '--videos-dir', config['dataset_root'],
            '--output-dir', str(Path(config['output'])/'runs'/job['label']),
            '--cache-dir', str(Path(config['output'])/'media_cache'),
            '--system-prompt', config['system_prompt'], '--agent-call-schema', 'task-start',
            '--agent-output-protocol', 'task-text']
    for key, value in config['inference'].items():
        args += ['--'+key.replace('_','-'), str(value)]
    return args + ['--gpus', *gpus]


def dry_run(config, selected, gpus):
    validate(config)
    result = []
    for job in selected:
        ready, reason = True, 'checkpoint_available'
        if job.get('training_output'):
            try:
                training_ready(job)
                ready = export_ready(job)
                reason = 'export_ready' if ready else 'final_checkpoint_ready_export_pending'
            except (ValueError, FileNotFoundError, KeyError):
                ready, reason = False, 'selected_training_final_not_ready'
        else:
            ready = (Path(job['model_path'])/'config.json').is_file()
            reason = 'checkpoint_available' if ready else 'checkpoint_unavailable'
        result.append(dict(model=job['label'], ready=ready, state=reason,
                           argv=command(config,job,gpus or ['<explicit-GPU-ID>'])))
    return result


def run(config_path, model, gpus=None, *, stage='inference', judge_base_url=None, check_only=False):
    config = json.loads(Path(config_path).read_text())
    selected = [m for m in config['models'] if model=='all' or m['label']==model]
    if not selected:
        raise ValueError('Unknown model label')
    if check_only:
        return dry_run(config,selected,gpus)
    validate(config)
    if stage=='inference' and (not gpus or len(gpus)!=len(set(gpus)) or any(not g.isdigit() for g in gpus)):
        raise ValueError('Inference requires explicit, distinct numeric GPU IDs')
    root = Path(config['output'])
    with (root/'.external.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stage=='judge':
            from benchmark.external.scoring import judge
            for job in selected:
                result = judge(config,job['label'],judge_base_url)
                report(config_path)
                if result['failed']:
                    raise RuntimeError('Judge has failed records; resume explicitly, never substitute zeros')
            return report(config_path)
        # Resolve all requested final-checkpoint dependencies before inference.
        for job in selected:
            if job.get('training_output'):
                ensure_export(config,job)
            elif not (Path(job['model_path'])/'config.json').is_file():
                raise ValueError('Base checkpoint unavailable')
        memory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        occupied = {parts[0].strip():int(parts[1]) for line in memory.splitlines() if (parts:=line.split(','))}
        if any(g not in occupied or occupied[g]>config['max_occupied_memory_mib'] for g in gpus):
            raise RuntimeError('Requested GPUs are unavailable; no existing process will be stopped')
        for job in selected:
            # Foreground supervisor owns this child; report periodically and reap
            # it on interruption. No detached process survives this command.
            process = subprocess.Popen(command(config,job,gpus),cwd=REPO,start_new_session=True)
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            previous_term = signal.signal(signal.SIGTERM,interrupted)
            try:
                while process.poll() is None:
                    report(config_path)
                    time.sleep(config['report_interval_seconds'])
                if process.returncode:
                    raise RuntimeError('Inference failed; complete rows remain resumable')
            finally:
                try:
                    if process.poll() is None:
                        from benchmark.runtime.parallel import terminate_owned
                        terminate_owned([process])
                    report(config_path)
                finally:
                    signal.signal(signal.SIGTERM,previous_term)
        return report(config_path)
