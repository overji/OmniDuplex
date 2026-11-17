"""Continue the 20261116 experiments after their existing jobs finish."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ROOT = Path('/root/jimodels/OmniDuplex')
CONTROL = ROOT / 'train/outputs/grpo_fp15_baseline_eval_queue_20261117'
PREVIOUS = ROOT / 'train/outputs/grpo_judge_fp1_resume16_followup_20261116'
SOURCE = Path('/data/OmniDuplex/training_outputs/grpo_standard_judge_fp1_kl003_20261116')
NEW = Path('/data/OmniDuplex/training_outputs/grpo_local_judge_fp15_kl003_20261117')
DEST = Path('/root/jidata/OmniDuplex/training_outputs/grpo_local_judge_fp15_kl003_20261117')
EXPORT = Path('/root/jidata/openmodels/OmniDuplex-MiniCPM-o-4.5-GRPO-standard-judge-fp1-kl003-step40-20261117')
RESULT = ROOT / 'benchmark/results/grpo_standard_judge_fp1_kl003_step40_aoi_20261117'
PYTHON = ROOT / 'train/.venv/bin/python'
state = {}
lock = threading.Lock()

def mark(**fields):
    with lock:
        state.update(fields, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid())
        tmp = CONTROL / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(tmp, CONTROL / 'status.json')

def prior():
    return json.loads((PREVIOUS / 'status.json').read_text())

def wait_prior(field):
    while True:
        s = prior()
        if s.get(field) == 'complete':
            return
        if s.get(field) == 'failed' or s.get('stage') == 'failed':
            raise RuntimeError(f'Previous {field} failed: {s}')
        time.sleep(30)

def wait_gpus(ids):
    while True:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        used = {int(a.strip()): int(b.strip()) for line in lines for a, b in [line.split(',')]}
        if all(used[i] < 2048 for i in ids):
            return
        time.sleep(30)

def run(name, command, env=None):
    with (CONTROL / (name + '.log')).open('ab', buffering=0) as log:
        p = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        mark(**{name: 'running', name + '_pid': p.pid})
        if p.wait():
            raise RuntimeError(f'{name} failed; inspect {name}.log')

def judge_ready():
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8917/v1/models', timeout=3) as response:
            return 'Qwen3.8-27B' in {x['id'] for x in json.load(response)['data']}
    except Exception:
        return False

def train():
    mark(fp15='waiting_for_baseline')
    wait_prior('baseline')
    mark(fp15='waiting_for_gpus')
    wait_gpus([0, 1, 2, 3, 7])
    if NEW.exists() or DEST.exists():
        raise RuntimeError('FP=1.5 output already exists; refusing to overwrite')
    if judge_ready():
        raise RuntimeError('Judge endpoint already occupied')
    judge = migrator = None
    try:
        with (CONTROL / 'judge.log').open('ab', buffering=0) as log:
            judge = subprocess.Popen([str(PYTHON), str(ROOT / 'train/scripts/serve_grpo_judge.py'), '--config', str(ROOT / 'train/configs/judge_qwen27b_service.json'), '--start'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        mark(fp15='waiting_for_judge', judge_pid=judge.pid)
        while not judge_ready():
            if judge.poll() is not None:
                raise RuntimeError('Judge exited during startup')
            time.sleep(15)
        env = dict(os.environ, MIGRATION_CONTROL_DIR=str(CONTROL), MIGRATION_SOURCE=str(NEW), MIGRATION_DEST=str(DEST))
        with (CONTROL / 'migration.log').open('ab', buffering=0) as log:
            migrator = subprocess.Popen([str(PYTHON), str(ROOT / 'train/outputs/grpo_judge_fp1_resume10_20261116/migrate.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        run('fp15', [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', PREVIOUS / 'entry.py', '--config', CONTROL / 'grpo.json', 'grpo', '--prepared', ROOT / 'train/prepared/grpo_kl003_120qa_20261115', '--output', NEW, '--max-steps', '40'], dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1'))
        for i in range(4):
            report = json.loads((NEW / f'memory_report.rank{i}.json').read_text())
            if not report.get('checkpoint_saved') or report.get('optimizer_steps') != 40:
                raise RuntimeError(f'Training incomplete on rank {i}')
        (CONTROL / 'TRAIN_DONE').touch()
        mark(fp15='waiting_for_checkpoint_migration')
        if migrator.wait():
            raise RuntimeError('Checkpoint migration failed')
        mark(fp15='complete', fp15_checkpoint=str(DEST / 'checkpoint-final'))
    finally:
        if migrator and migrator.poll() is None and not (CONTROL / 'TRAIN_DONE').exists():
            (CONTROL / 'TRAIN_FAILED').touch()
        if judge and judge.poll() is None:
            judge.terminate()

def evaluate():
    mark(baseline_eval='waiting_for_baseline_and_current_benchmark')
    wait_prior('baseline')
    wait_prior('evaluation')
    if not (SOURCE / 'checkpoint-final/training_config.json').is_file():
        raise RuntimeError('Baseline final checkpoint missing')
    if EXPORT.exists() or RESULT.exists():
        raise RuntimeError('Baseline export or evaluation output exists; refusing to overwrite')
    run('baseline_export', [PYTHON, '-m', 'train', 'export', '--checkpoint', SOURCE / 'checkpoint-final', '--output', EXPORT], dict(os.environ, PYTHONUNBUFFERED='1'))
    if not (EXPORT / 'full_training_model.json').is_file():
        raise RuntimeError('Missing model export completion marker')
    mark(baseline_eval='waiting_for_gpus')
    wait_gpus([4, 5, 6])
    q = json.loads((ROOT / 'train/outputs/sequential_sft_grpo_full_20261114/queue.json').read_text())
    cmd = q[0]['command'][:]
    for flag, value in [('--model-path', EXPORT), ('--output-dir', RESULT), ('--sample-indices', ROOT / 'train/outputs/grpo_kl003_gpu_monitor_60_20261115/aoi_shuffle.json')]:
        cmd[cmd.index(flag) + 1] = str(value)
    cmd = cmd[:cmd.index('--gpus')] + ['--gpus', '4', '5', '6']
    run('baseline_eval', cmd, dict(os.environ, PYTHONUNBUFFERED='1'))
    mark(baseline_eval='complete', baseline_eval_output=str(RESULT))

def main():
    os.chdir(ROOT)
    CONTROL.mkdir(parents=True, exist_ok=True)
    mark(stage='running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train): 'fp15', executor.submit(evaluate): 'baseline_eval'}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                mark(**{name: 'failed', name + '_error': repr(exc)})
    mark(stage='complete' if state.get('fp15') == state.get('baseline_eval') == 'complete' else 'finished_with_errors')

if __name__ == '__main__':
    main()
