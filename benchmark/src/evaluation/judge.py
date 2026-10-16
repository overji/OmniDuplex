"""Resumable, reference-based content judging of existing timing matches."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import time

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.evaluation.agent_calls import parse_event
from benchmark.evaluation.timing import match_windows, reference_windows
from benchmark.evaluation.judge_client import completion, resolve_config

VERSION = 'reference_content_v1'
SCORES = (0, 0.5, 1, 1.5, 2)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_answer(text, judge_call):
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Expected JSON object')
    for key in ('response_score', 'call_score'):
        score = obj.get(key)
        required = key == 'response_score' or judge_call
        if required:
            if type(score) not in (int, float) or score not in SCORES:
                raise ValueError('Invalid score')
        elif score is not None:
            raise ValueError('Unexpected call score')
    if not judge_call and 'call_reason' in obj and obj['call_reason'] is None:
        obj['call_reason'] = 'Not applicable'
    for key in ('response_reason', 'call_reason'):
        if not isinstance(obj.get(key), str):
            raise ValueError('Missing reason')
    if not isinstance(obj.get('error_types'), list) or any(not isinstance(x, str) for x in obj['error_types']):
        raise ValueError('Invalid error types')
    return {key: obj[key] for key in ('response_score', 'call_score', 'response_reason', 'call_reason', 'error_types')}


def prepare_rows(samples, predictions):
    rows = []
    seen = set()
    negative_samples = []
    for prediction in predictions:
        index = prediction['sample_index']
        if type(index) is not int or not 0 <= index < len(samples) or index in seen:
            raise ValueError('Invalid or duplicate sample_index')
        seen.add(index)
        sample = samples[index]
        for key in ('task', 'video', 'question'):
            if prediction.get(key) != sample.get(key):
                raise ValueError(f'Prediction/data mismatch: {index} {key}')
        if prediction.get('status') != 'complete':
            continue
        events = prediction['events']
        for event in events:
            onset = event.get('onset_seconds')
            if type(onset) not in (int, float) or not math.isfinite(onset) or not isinstance(event.get('text'), str):
                raise ValueError('Invalid prediction event')
        kind = sample.get('sample_type', sample.get('generation_context', {}).get('sample_type'))
        agent = sample['mode'] == 'agent'
        positive = agent and kind not in ('missing_tool', 'no_context')
        parsed = [parse_event(event) for event in events]
        if agent and not positive:
            negative_samples.append({'sample_index': index, 'sample_type': kind,
                'erroneous_call': any(p['attempted'] for p in parsed)})
        matched = {m['response_index']: m['event_index'] for m in match_windows(sample, events)['matches']}
        for ri, reference in reference_windows(sample):
            ei = matched.get(ri)
            candidate = parsed[ei] if ei is not None else {'response': '', 'calls': [], 'status': 'unmatched'}
            actual_call = positive and candidate['status'] != 'malformed' and bool(candidate['calls'])
            # Whitelist only the current reference. Never send later tool results or completion replies.
            ref = {key: reference[key] for key in ('response', 'tool_calls', 'evidence', 'details',
                'trigger', 'refusal_reason', 'missing_tools') if key in reference}
            payload = {'question': sample['question'], 'question_time': sample.get('question_time'),
                'mode': sample['mode'], 'ability': sample.get('ability'), 'sample_type': kind,
                'reference': ref, 'candidate_response': candidate['response'],
                'candidate_tool_calls': candidate['calls'], 'judge_call': actual_call}
            if agent:
                payload.update(available_tools=sample.get('available_tools', []),
                    frontend_context=sample.get('frontend_context', {}))
            row = {'sample_index': index, 'response_index': ri, 'event_index': ei,
                'mode': sample['mode'], 'ability': sample.get('ability'), 'sample_type': kind,
                'call_applicable': positive, 'parse_status': candidate['status'], 'payload': payload}
            rows.append(row)
    return rows, negative_samples


def evaluate(row, config, key, system, request=completion):
    record = {k: v for k, v in row.items() if k != 'payload'}
    record['usage_attempts'] = []
    payload = row['payload']
    if row['event_index'] is None or (not payload['candidate_response'].strip() and not payload['judge_call']):
        return {**record, 'status': 'complete', 'response_score': 0,
            'call_score': 0 if row['call_applicable'] else None,
            'response_reason': 'No matched response' if row['event_index'] is None else 'Empty response',
            'call_reason': 'No valid call' if row['call_applicable'] else 'Not applicable',
            'error_types': ['missing_output']}
    for attempt in range(config['retries'] + 1):
        usage = {}
        try:
            text, usage, complete = request(config, key, system, payload)
            record['usage_attempts'].append({'attempt': attempt + 1, 'usage': usage,
                                             'raw_response': text, 'response_complete': complete})
            if not complete:
                raise ValueError('Truncated or blocked judge output')
            answer = validate_answer(text, payload['judge_call'])
            if not payload['candidate_response'].strip():
                answer.update(response_score=0, response_reason='Empty response')
            if row['call_applicable'] and not payload['judge_call']:
                answer.update(call_score=0, call_reason='Missing or malformed actual call')
            return {**record, **answer, 'status': 'complete'}
        except Exception as exc:
            # Do not persist exception messages: providers can echo credentials or requests.
            if len(record['usage_attempts']) <= attempt:
                record['usage_attempts'].append({'attempt': attempt + 1, 'usage': usage})
            record['usage_attempts'][-1]['error_type'] = type(exc).__name__
            if attempt < config['retries']:
                time.sleep(min(2 ** attempt, 8))
    return {**record, 'status': 'failed', 'response_score': None, 'call_score': None}


def summarize(rows, negatives, total_samples, evaluated_samples):
    groups = defaultdict(list)
    for row in rows:
        names = ['overall', 'mode:' + row['mode'], 'ability:' + str(row['ability'])]
        if row['mode'] == 'agent' and row['sample_type']:
            names.append('agent:' + row['sample_type'])
        for name in names:
            groups[name].append(row)
    result = {}
    for name, items in groups.items():
        completed = [r for r in items if r['status'] == 'complete']
        matched = [r for r in items if r['event_index'] is not None]
        matched_ok = [r for r in completed if r['event_index'] is not None]
        calls = [r for r in items if r['call_applicable']]
        calls_ok = [r for r in completed if r['call_applicable']]
        def mean(values):
            return sum(values) / len(values) if values else None
        result[name] = {'windows': len(items), 'scored_windows': len(completed),
            'failed_windows': len(items) - len(completed), 'matched_windows': len(matched),
            'response_mean_matched': mean([r['response_score'] for r in matched_ok]) if len(matched_ok) == len(matched) else None,
            'response_mean_all_windows': mean([r['response_score'] for r in completed]) if len(completed) == len(items) else None,
            'call_windows': len(calls),
            'call_mean_all_windows': mean([r['call_score'] for r in calls_ok]) if len(calls_ok) == len(calls) else None,
            'call_mean_matched': mean([r['call_score'] for r in calls_ok if r['event_index'] is not None])
                if all(r['status'] == 'complete' for r in calls if r['event_index'] is not None) else None}
    return {'evaluation': VERSION, 'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'incomplete',
        'total_benchmark_samples': total_samples, 'evaluated_samples': evaluated_samples,
        'excluded_or_missing_samples': total_samples - evaluated_samples,
        'groups': result, 'negative_agent': {'samples': len(negatives),
            'erroneous_calls': sum(n['erroneous_call'] for n in negatives),
            'error_rate': sum(n['erroneous_call'] for n in negatives) / len(negatives) if negatives else None,
            'per_sample': negatives},
        'note': 'Text/reference consistency only. Legacy prompts did not require structured calls; missing positive calls score zero.'}


def write_report(path, metrics):
    lines = ['# Content judge', '', 'Status: ' + metrics['status'], '',
        f"Completed inference samples: {metrics['evaluated_samples']}/{metrics['total_benchmark_samples']}", '', metrics['note'], '',
        '| Group | Scored / windows | Matched response mean | All-window response mean | All-window call mean |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, group in metrics['groups'].items():
        fmt = lambda value: 'N/A' if value is None else f'{value:.3f}'
        lines.append(f"| {name} | {group['scored_windows']}/{group['windows']} | " + ' | '.join(fmt(group[k]) for k in (
            'response_mean_matched', 'response_mean_all_windows', 'call_mean_all_windows')) + ' |')
    lines.extend(['', 'Negative Agent erroneous-call rate: ' + str(metrics['negative_agent']['error_rate'])])
    from benchmark.data.files import atomic_write_text
    atomic_write_text(path, '\n'.join(lines) + '\n')


def execute(args):
    if args.concurrency < 1 or args.timeout <= 0 or args.retries < 0:
        raise ValueError('Invalid concurrency, timeout or retries')
    if len({r.resolve() for r in args.runs}) != len(args.runs) or len({r.name for r in args.runs}) != len(args.runs):
        raise ValueError('Run directories must be unique and have distinct names')
    key, url = resolve_config(args)
    system = args.system_prompt.read_text()
    samples = json.loads(args.data.read_text())
    config = {'version': VERSION, 'provider': args.provider, 'model': args.model,
        'base_url': url, 'timeout': args.timeout, 'retries': args.retries, 'temperature': 0,
        'system_prompt_sha256': fingerprint(system), 'data_sha256': fingerprint(samples)}
    summaries = []
    for run in args.runs:
        predictions = read_jsonl(run / 'predictions.jsonl')
        if not (run / 'predictions.jsonl').is_file():
            raise FileNotFoundError(run / 'predictions.jsonl')
        rows, negatives = prepare_rows(samples, predictions)
        identity = {**config, 'predictions_sha256': fingerprint(predictions)}
        out = args.output_dir / run.name
        out.mkdir(parents=True, exist_ok=True)
        with (out / '.judge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config_path = out / 'judge_config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != identity:
                raise ValueError('Judge inputs/config changed; choose a new output directory')
            if not config_path.exists() and (out / 'judgments.jsonl').exists():
                raise ValueError('Existing judgments have no config; choose a new output directory')
            write_json(config_path, identity)
            history_path = out / 'judgments.jsonl'
            recover_jsonl_tail(history_path)
            history = read_jsonl(history_path)
            cache = {r['cache_key']: r for r in history if r['status'] == 'complete'}
            results = []
            pending = []
            for row in rows:
                row['cache_key'] = fingerprint({'config': identity, 'row': row})
                if row['cache_key'] in cache:
                    results.append(cache[row['cache_key']])
                else:
                    pending.append(row)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(evaluate, row, config, key, system) for row in pending]
                for future in as_completed(futures):
                    record = future.result()
                    append_jsonl(history_path, record)
                    history.append(record)
                    results.append(record)
                    print(f'{run.name}: {len(results)}/{len(rows)} ({record["status"]})', flush=True)
            results.sort(key=lambda r: (r['sample_index'], r['response_index']))
            metrics = summarize(results, negatives, len(samples), sum(p.get('status') == 'complete' for p in predictions))
            write_json(out / 'judge_metrics.json', metrics)
            write_report(out / 'judge_report.md', metrics)
            attempts = [a for r in history for a in r['usage_attempts']]
            tokens = defaultdict(float)
            for a in attempts:
                for name, value in a['usage'].items():
                    if type(value) in (int, float):
                        tokens[name] += value
            write_json(out / 'api_usage.json', {'requests': len(attempts),
                'failed_attempts': sum('error_type' in a for a in attempts), 'usage_totals': dict(tokens)})
            summaries.append({'model': run.name, 'status': metrics['status'], **metrics['groups'].get('overall', {})})
    write_json(args.output_dir / 'judge_comparison.json', summaries)
    import csv
    from io import StringIO
    from benchmark.data.files import atomic_write_text
    stream = StringIO()
    fields = list(dict.fromkeys(k for s in summaries for k in s))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summaries)
    atomic_write_text(args.output_dir / 'judge_comparison.csv', stream.getvalue())
    if any(s['status'] != 'complete' for s in summaries):
        raise SystemExit(1)
