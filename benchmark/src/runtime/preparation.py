"""CPU-only preparation: inventory and manifests, never adapter/model execution."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

from benchmark.config.paths import ROOT
from benchmark.data.files import write_json


def absolute(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT.parent / path


def checkpoint_status(path):
    config = path / ('thinker/config.json' if (path / 'thinker').is_dir() else 'config.json')
    issues = [] if config.is_file() else ['missing config.json']
    for index in path.rglob('*.index.json'):
        if '.cache' in index.parts:
            continue
        for name in set(json.loads(index.read_text()).get('weight_map', {}).values()):
            shard = index.parent / name
            if not shard.is_file() or not shard.stat().st_size:
                issues.append(f'missing/empty shard: {shard}')
    weights = [p for p in path.rglob('*.safetensors') if '.cache' not in p.parts]
    if not weights:
        issues.append('no safetensors weights')
    return {'path': str(path), 'weight_files': len(weights),
            'weight_bytes': sum(p.stat().st_size for p in weights), 'issues': issues}


def environment_status(python, modules, source=None, checkpoint=None, class_reference=None):
    if not python.is_file():
        return {'ready': False, 'error': 'interpreter missing', 'python': str(python)}
    script = """import importlib,json,sys
sys.path.insert(0,sys.argv[1])
loaded={}
for spec in sys.argv[4:]:
    name,_,attribute=spec.partition(':')
    module=importlib.import_module(name)
    if attribute: getattr(module,attribute)
    loaded[spec]=str(getattr(module,'__version__','importable'))
if sys.argv[3]:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    get_class_from_dynamic_module(sys.argv[3],sys.argv[2],local_files_only=True)
    loaded['native_model_class']='importable; no weights loaded'
print(json.dumps(loaded))
"""
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1'}
    source = source / 'minicpm_ft' if source and (source / 'minicpm_ft').is_dir() else source
    try:
        result = subprocess.run([str(python), '-c', script, str(source or ''), str(checkpoint or ''), class_reference or '', *modules],
                                capture_output=True, text=True, env=env, timeout=60)
        return {'ready': result.returncode == 0, 'python': str(python),
                'imports': result.stdout[-2000:], 'error': result.stderr[-2000:] if result.returncode else ''}
    except subprocess.TimeoutExpired:
        return {'ready': False, 'python': str(python), 'error': 'CPU import timeout'}


def media_aliases(rows, videos, metadata, output):
    manifest = json.loads(metadata.read_text()) if metadata.is_file() else {}
    records = ([{'file': Path(r['path']).name} for r in manifest]
               if isinstance(manifest, list) else manifest.get('videos', []))
    known_files = {r['file'] for r in records}
    originals = {}
    for r in records:
        if r.get('original_file'):
            originals.setdefault(Path(r['original_file']).name, set()).add(r['file'])
    output.mkdir(parents=True, exist_ok=True)
    missing, aliases = [], {}
    for name in sorted({str(row['video']) for row in rows}):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe video path: {name}')
        source = videos / relative
        if not source.is_file() and relative.name in known_files and (videos / relative.name).is_file():
            source = videos / relative.name
        if not source.is_file():
            matches = [videos / n for n in originals.get(relative.name, ()) if (videos / n).is_file()]
            if len(matches) > 1:
                raise ValueError(f'Ambiguous original video name: {name}')
            source = matches[0] if matches else source
        if not source.is_file():
            missing.append(name)
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f'Existing media alias changed: {target}')
        else:
            target.symlink_to(source.resolve())
        aliases[name] = str(source.resolve())
    return {'aliases': aliases, 'missing_videos': missing,
            'missing_qa': sum(str(r['video']) in missing for r in rows),
            'unique_resolved_media': len(set(aliases.values()))}


def execute(args):
    suite = json.loads(args.suite.read_text())
    data = absolute(suite['data'])
    rows = json.loads(data.read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    aliases = out / 'videos'
    report = {'suite': suite['name'], 'source_qa': len(rows), 'inference_started': False,
              'gpu_models_loaded': False, 'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
              'media': media_aliases(rows, absolute(suite['videos_dir']), absolute(suite['video_metadata']), aliases),
              'models': []}
    commands, environments = [], {}
    for item in suite['models']:
        path = Path(item['checkpoint'])
        path = path if path.is_absolute() else Path(suite['model_root']) / path
        python = absolute(item['python'])
        source = absolute(item['official_code']) if item.get('official_code') else None
        class_ref = item.get('class_reference')
        model_alias = out / 'model_aliases' / ('model_' + hashlib.sha256(str(path).encode()).hexdigest()[:16])
        if class_ref:
            model_alias.parent.mkdir(parents=True, exist_ok=True)
            if not model_alias.exists():
                model_alias.symlink_to(path.resolve(), target_is_directory=True)
        key = (str(python), str(source), tuple(item['modules']), class_ref, str(path))
        if key not in environments:
            environments[key] = environment_status(python, item['modules'], source, model_alias, class_ref)
        weights = checkpoint_status(path)
        ready = not weights['issues'] and environments[key]['ready'] and (not source or source.is_dir())
        report['models'].append({'mode': item['mode'], 'adapter': item['adapter'], 'cpu_prepared': ready,
                                'checkpoint': weights, 'environment': environments[key]})
        cmd = [str(python), '-m', 'benchmark']
        if item['mode'] == 'probe':
            cmd += ['probe', 'run', item['adapter'], '--manifest', str(out / 'probe/probe_manifest.jsonl')]
        else:
            cmd += ['run', item['adapter'], '--data', str(data), '--videos-dir', str(aliases)]
        cmd += ['--model-path', str(path), '--output-dir', str(ROOT / 'results' / suite['name'] / item['mode'] / item['adapter'])]
        if source and (item['mode'] == 'probe' or item['adapter'] in {'gander', 'livestar'}):
            cmd += ['--official-code', str(source)]
        for flag, value in item.get('arguments', {}).items():
            cmd += ['--' + flag, str(value)]
        commands.append({'mode': item['mode'], 'adapter': item['adapter'], 'gpu_assignment': 'required_at_launch',
                         'service_required': item.get('service_required', False), 'argv': cmd, 'shell': shlex.join(cmd)})
        print(f"PREPARE {item['mode']}/{item['adapter']}: {'CPU READY' if ready else 'INCOMPLETE'}", flush=True)
    if not args.skip_probes:
        from benchmark.probe.data import file_hash, prepare
        target = out / 'probe'
        config = target / 'probe_manifest_config.json'
        if config.is_file():
            old = json.loads(config.read_text())
            if old['data_sha256'] != file_hash(data) or old['videos_dir'] != str(aliases):
                raise ValueError('Existing probe manifest differs; use a new output directory')
        else:
            prepare(SimpleNamespace(data=data, videos_dir=aliases, output_dir=target))
        report['probe'] = json.loads((target / 'probe_prepare_report.json').read_text())
    report['all_cpu_prepared'] = all(m['cpu_prepared'] for m in report['models'])
    report['gpu_inference_verified'] = False
    if 'probe' not in report and (out / 'probe/probe_prepare_report.json').is_file():
        report['probe'] = json.loads((out / 'probe/probe_prepare_report.json').read_text())
    service = next((m for m in suite['models'] if m['adapter'] == 'joyai'), None)
    if service:
        service_config = {'code': str(absolute(service['official_code'])),
            'python': str(absolute(service['service_python'])),
            'main_model_path': str(Path(suite['model_root']) / service['checkpoint']),
            'summary_model_path': str(Path(suite['model_root']) / 'Qwen3-VL-4B-Instruct')}
        write_json(out / 'joyai_services.json', service_config)
        report['joyai_service_config'] = str(out / 'joyai_services.json')
        report['joyai_service_interpreter_exists'] = Path(service_config['python']).is_file()
    write_json(out / 'run_commands.json', commands)
    write_json(out / 'preparation_report.json', report)
    lines = ['# Benchmark preparation status', '', 'No model inference or GPU model loading was performed.', '',
             f"QA: {len(rows)}; missing media: {len(report['media']['missing_videos'])} videos / {report['media']['missing_qa']} QA.", '',
             '| Mode | Model | CPU preparation |', '| --- | --- | --- |']
    lines += [f"| {m['mode']} | {m['adapter']} | {'ready' if m['cpu_prepared'] else 'incomplete (see JSON)'} |" for m in report['models']]
    (out / 'preparation_report.md').write_text('\n'.join(lines) + '\n')
    print(f'Preparation report: {out / "preparation_report.md"}', flush=True)
