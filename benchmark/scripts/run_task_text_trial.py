#!/usr/bin/env python3
"""Supervise a matched Agent text-format trial, then resume the paused full suite."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.data.files import read_jsonl, write_json, atomic_write_text
from benchmark.data.agent_inputs import input_protocol
from benchmark.data.agent_schema import validate_task_start
from benchmark.evaluation.agent_calls import parse_event
from benchmark.evaluation.timing import summarize, match_windows
from benchmark.runtime.parallel import terminate_owned
from benchmark.scripts.run_prepared_suite import option


def absolute(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def stratified_indices(samples, eligible, count, seed):
    groups = defaultdict(list)
    for i in sorted(eligible):
        groups[samples[i]['ability']].append(i)
    if not 0 < count <= len(eligible):
        raise ValueError('Not enough shared completed Agent QA')
    quotas = {k: count * len(v) / len(eligible) for k, v in groups.items()}
    sizes = {k: int(v) for k, v in quotas.items()}
    remainder = count - sum(sizes.values())
    for k in sorted(groups, key=lambda k: (-(quotas[k] - sizes[k]), k))[:remainder]:
        sizes[k] += 1
    rng = random.Random(seed)
    selected = []
    for k in sorted(groups):
        rng.shuffle(groups[k])
        selected.extend(groups[k][:sizes[k]])
    return sorted(selected)


def completed(directory):
    return {r['sample_index']: r for r in read_jsonl(directory / 'predictions.jsonl', complete_only=True)
            if r.get('status') == 'complete'}


def snapshot_report(root, samples, plan):
    result = {'selected_samples': len(plan['indices']), 'models': {}}
    lines = ['# Agent 纯文本分隔符试验', '',
             '模型输出：`回复<|task_start|>instruction`。格式合规不代表 instruction 语义正确或原生工具执行成功。',
             '仅使用已完成 QA；未调用付费内容 judge。JoyAI 为视觉输入，其他两者为音视频输入。', '',
             '| 模型 | 版本 | 完成 QA | 输出事件 | Recall | Precision | F1 | 双字段合规 QA |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in plan['models']:
        base = completed(root / 'baseline' / name)
        merged = {}
        for path in [root / name / 'smoke', root / name / 'full']:
            for i, row in completed(path).items():
                if i in merged:
                    raise ValueError('Smoke/full sample duplicated')
                merged[i] = row
        directory = root / name
        directory.mkdir(exist_ok=True)
        atomic_write_text(directory / 'predictions.jsonl', ''.join(json.dumps(merged[i], ensure_ascii=False) + '\n' for i in sorted(merged)))
        both = {}
        for version, rows in [('baseline', base), ('task-text', merged)]:
            metrics = summarize(samples, list(rows.values()))
            score = metrics['groups'].get('overall', {})
            counts = Counter()
            call_events = []
            for i, row in rows.items():
                parsed = [parse_event(e) for e in row['events']]
                counts['qa_with_parseable_calls'] += any(p['calls'] for p in parsed)
                counts['qa_with_schema_valid_calls'] += any(p['calls'] and p['status'] != 'malformed' and all(validate_task_start(c) for c in p['calls']) for p in parsed)
                counts['qa_with_separator'] += any(e.get('task_text_separator_count', 0) for e in row['events'])
                counts['qa_with_valid_dual_field'] += any(e.get('task_text_format_valid', False) and e.get('task_text_generation_complete') is not False for e in row['events'])
                for e in row['events']:
                    if e.get('task_text_status'):
                        counts['event_status:' + e['task_text_status']] += 1
                timed = [{'onset_seconds': e['agent_call_onset_seconds']} for e in row['events']
                         if e.get('tool_calls') and e.get('agent_call_onset_seconds') is not None]
                counts['call_time_window_hits'] += match_windows(samples[i], timed)['hits']
            both[version] = {'metrics': metrics, 'format': dict(counts)}
            if score:
                dual = str(counts['qa_with_valid_dual_field']) if version == 'task-text' else '不适用（JSON/native）'
                lines.append(f'| {name} | {version} | {score["samples"]} | {score["predictions"]} | {score["window_hit_rate"]:.2%} | {score["timing_precision"]:.2%} | {score["timing_f1"]:.2%} | {dual} |')
        result['models'][name] = both
        write_json(directory / 'metrics.json', both['task-text']['metrics'])
        write_json(directory / 'format_metrics.json', both['task-text']['format'])
    lines += ['', '未完成前，两版本覆盖率不同，不将其直接作为最终公平比较；全部完成后两版本使用完全相同的 100 条。', '', '## 原始输出示例', '']
    for name in plan['models']:
        lines += [f'### {name}', '']
        rows = completed(root / name)
        shown = set()
        for i in sorted(rows):
            ability = samples[i]['ability']
            if ability in shown:
                continue
            shown.add(ability)
            lines += [f'- sample {i} / {ability}: {samples[i]["question"]}',
                      '```text', '\n'.join(e['text'] for e in rows[i]['events']) or '(no response)', '```', '']
    atomic_write_text(root / 'report.md', '\n'.join(lines) + '\n')
    write_json(root / 'comparison.json', result)


def prepare(config, root):
    entries = json.loads(absolute(config['commands']).read_text())
    lookup = {e['adapter']: e for e in entries if e['mode'] == 'realtime'}
    argv = lookup['minicpmo']['argv']
    data = Path(argv[argv.index('--data') + 1]);videos = Path(argv[argv.index('--videos-dir') + 1])
    samples = json.loads(data.read_text())
    baselines = {n: completed(absolute(config['baselines'][n])) for n in config['models']}
    eligible = set.intersection(*(set(v) for v in baselines.values()))
    eligible = {i for i in eligible if samples[i]['mode'] == 'agent'
                and samples[i].get('sample_type', samples[i].get('generation_context', {}).get('sample_type')) == 'positive'
                and (videos / samples[i]['video']).is_file()}
    selected = stratified_indices(samples, eligible, config['sample_count'], config['seed'])
    root.mkdir(parents=True, exist_ok=True)
    from types import SimpleNamespace
    protocol = input_protocol(SimpleNamespace(agent_output_protocol='task-text', agent_call_schema='task-start'))
    queue_root = absolute(config['queue_root'])
    old_commands = json.loads((queue_root / 'remaining_commands.json').read_text())
    start = next(i for i, e in enumerate(old_commands) if e['mode'] == 'realtime' and e['adapter'] == 'livestar')
    remaining = old_commands[start:]
    plan = {'models': config['models'], 'gpus': config['gpus'], 'indices': selected,
            'seed': config['seed'], 'eligible_shared_agent': len(eligible), 'data': str(data),
            'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(), 'protocol': protocol,
            'ability_counts': dict(Counter(samples[i]['ability'] for i in selected)),
            'queue_root': str(queue_root), 'baselines': config['baselines'], 'commands': {}}
    for name in config['models']:
        command = lookup[name]['argv']
        command = list(command);command[command.index('run')] = 'parallel'
        command = option(command, '--agent-call-schema', 'task-start')
        command = option(command, '--agent-output-protocol', 'task-text')
        command = option(command, '--sample-indices', root / 'indices.json')
        if name == 'joyai':
            command += ['--joyai-services', str(absolute(config['joyai_services']))]
        plan['commands'][name] = command
    destination = root / 'plan.json'
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError('Trial plan changed; use a new output directory')
    write_json(destination, plan)
    write_json(root / 'indices.json', selected)
    write_json(root / 'remaining_indices.json', selected[1:])
    extension = root / 'full_agent_schedule.json'
    if extension.is_file():
        extra = json.loads(extension.read_text())['commands']
        if [e['adapter'] for e in extra] != config['models']:
            raise ValueError('Full Agent extension has unexpected models')
        remaining = [{**e, 'sequence': i} for i, e in enumerate([*extra, *remaining])]
    write_json(root / 'continuation_commands.json', remaining)
    for name in config['models']:
        folder = root / 'baseline' / name;folder.mkdir(parents=True, exist_ok=True)
        atomic_write_text(folder / 'predictions.jsonl', ''.join(json.dumps(baselines[name][i], ensure_ascii=False) + '\n' for i in selected))
    prompts = root / 'prompts';prompts.mkdir(exist_ok=True)
    from benchmark.data.task_text import build_prompt
    for name in config['models']:
        atomic_write_text(prompts / f'{name}_sample_{selected[0]}.txt', build_prompt(samples[selected[0]], name))
    snapshot_report(root, samples, plan)
    return samples, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'benchmark/configs/suites/task_text_agent100.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args();config = json.loads(args.config.read_text());root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.trial.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    samples, plan = prepare(config, root)
    if args.dry_run:
        print(json.dumps({k: plan[k] for k in ['indices', 'ability_counts', 'eligible_shared_agent', 'gpus']}, indent=2))
        return
    # The user explicitly requests pausing the old queue before this supervisor starts.
    old_status = json.loads((absolute(config['queue_root']) / 'suite/suite_status.json').read_text())
    if old_status['status'] == 'running':
        raise RuntimeError('Pause the owned existing suite before starting the trial')
    processes = [];status = {'pid': os.getpid(), 'status': 'running', 'stage': 'agent_trials', 'jobs': []}
    def save():write_json(root / 'status.json', status)
    def interrupted(signum, frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted);signal.signal(signal.SIGINT, interrupted)
    def run_child(command, log_path, job):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('ab') as log:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED':'1', 'OMP_NUM_THREADS':'4', 'HF_HUB_OFFLINE':'1', 'CUDA_DEVICE_ORDER':'PCI_BUS_ID'})
            processes.append(child)
            job.update({'child_pid' if job is status else 'pid': child.pid});save()
            while child.poll() is None:
                snapshot_report(root, samples, plan);time.sleep(20)
            processes.remove(child)
            return child.returncode
    try:
        save()
        for name in plan['models']:
            job = {'model': name, 'status': 'smoke'};status['jobs'].append(job);save()
            command = plan['commands'][name]
            smoke = option(command, '--output-dir', root / name / 'smoke') + ['--gpus', plan['gpus'][0], '--max-samples', '1']
            try:
                code = run_child(smoke, root / name / 'smoke.log', job)
                if name == 'joyai' and code:
                    job['status'] = 'smoke_retrying';save()
                    code = run_child(smoke, root / name / 'smoke_retry.log', job)
                if code:
                    job.update(status='smoke_failed', exit_code=code);save();continue
                if len(completed(root / name / 'smoke')) != 1:
                    raise RuntimeError('Smoke did not produce exactly one complete QA')
                full = option(command, '--sample-indices', root / 'remaining_indices.json')
                full = option(full, '--output-dir', root / name / 'full') + ['--gpus', *plan['gpus']]
                job['status'] = 'running';save()
                code = run_child(full, root / name / 'full.log', job)
                if name == 'joyai' and code:
                    job['status'] = 'retrying';save()
                    code = run_child(full, root / name / 'retry.log', job)
                job.update(status='complete' if code == 0 and len(completed(root / name / 'full')) == len(plan['indices']) - 1 else 'incomplete', exit_code=code)
            except Exception as exc:
                job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            snapshot_report(root, samples, plan);save()
        status['stage'] = 'continuation';save()
        command = [sys.executable, str(REPO / 'benchmark/scripts/run_prepared_suite.py'),
            '--commands', str(root / 'continuation_commands.json'), '--output-dir', str(root / 'continuation'),
            '--reuse-root', str(absolute(config['queue_root']) / 'suite'), '--gpus', *plan['gpus']]
        code = run_child(command, root / 'continuation.log', status)
        status.update(status='complete' if code == 0 and all(j['status'] == 'complete' for j in status['jobs']) else 'incomplete', exit_code=code)
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
    finally:
        terminate_owned(processes);snapshot_report(root, samples, plan);save()


if __name__ == '__main__':
    main()
