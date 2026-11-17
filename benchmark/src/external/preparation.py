"""Prepare immutable cohorts and commands; no inference, export, Judge or queue insertion."""
import hashlib
import json
from pathlib import Path
import shutil
from copy import deepcopy

from benchmark.data.files import write_json
from benchmark.data.interactscope import digest, prepare, VERSION

ROOT = Path(__file__).resolve().parents[2]


def implementation_identity():
    paths = [*Path(__file__).parent.glob('*.py'), ROOT/'src/data/interactscope.py',
             ROOT/'src/adapters/minicpmo.py', ROOT/'src/runtime/runner.py',
             ROOT/'src/evaluation/timing.py', ROOT/'src/evaluation/judge.py',
             ROOT/'src/evaluation/scored_f1.py', ROOT/'src/data/task_text.py',
             ROOT/'configs/prompts/interactscope.txt']
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(paths))).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config['benchmark'] != 'interactscope':
        raise ValueError('Only Omni-InteractScope is supported')
    return config


def prepare_suite(path, output, *, download=False, public_videos=None, skip_media_sha256=False):
    config = read_config(path)
    if skip_media_sha256:
        config['media_verification'] = 'decode'
    output = Path(output).resolve()
    if (output/'experiment.json').exists():
        previous = json.loads((output/'experiment.json').read_text())
        if (previous['implementation_sha256'] != implementation_identity()
                or json.loads((output/'source_config.json').read_text()) != config):
            raise ValueError('Prepared protocol/configuration changed; use a fresh output directory')
    if download:
        from benchmark.external.download import download as fetch
        fetch(config)
    import subprocess
    if subprocess.check_output(['git', '-C', config['official_code'], 'rev-parse', 'HEAD'], text=True).strip() != config['official_code_revision']:
        raise ValueError('Official code revision changed')
    subprocess.run(['git','-C',config['official_code'],'diff','--quiet','HEAD','--',
                    'configs/tools.yaml','scripts/materialize_public_videos.py'],check=True)
    if public_videos:
        import sys
        subprocess.run([sys.executable,str(Path(config['official_code'])/'scripts/materialize_public_videos.py'),
            '--source',str(Path(public_videos).resolve()),'--dataset',config['dataset_root'],'--checksums'],check=True)
    result = prepare(config, output/'inputs')
    cfg = dict(deepcopy(config), output=str(output), data=str(output/'inputs/runnable_qa.json'),
               data_sha256=result['data_sha256'], source_sha256=result['source_sha256'],
               media_manifest_sha256=result['media_manifest_sha256'],
               protocol=VERSION, implementation_sha256=implementation_identity())
    for job in cfg['models']:
        if job.get('training_output'):
            job['export_ready_marker'] = str(output/'exports'/f'{job["label"]}_complete.json')
    target = output/'experiment.json'
    if target.exists() and json.loads(target.read_text()) != cfg:
        raise ValueError('Prepared configuration changed; use a new output directory')
    write_json(target, cfg)
    write_json(output/'source_config.json',config)
    commands = []
    for model in cfg['models']:
        commands.append(dict(label=model['label'], argv=[cfg['python'], '-m', 'benchmark', 'external', 'run',
            '--config', str(target), '--model', model['label'], '--gpus', '<GPU IDs: supply explicitly>'],
            ready='base_checkpoint_available' if model['label']=='base' else 'requires_verified_checkpoint_final_export'))
    commands.append(dict(label='text_judge', argv=[cfg['python'], '-m', 'benchmark', 'external', 'run',
        '--config', str(target), '--model', 'all', '--stage', 'judge', '--judge-base-url', '<local Qwen27B endpoint>']))
    commands.append(dict(label='report', argv=[cfg['python'], '-m', 'benchmark', 'external', 'report', '--config', str(target)]))
    write_json(output/'run_commands.json', commands)
    from benchmark.external.reporting import report
    report(target)
    return result
