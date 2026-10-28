"""Score a portable input on a CPU-only machine and return one auditable JSON."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.evaluation.judge import evaluate_content
from benchmark.evaluation.judge_client import resolve_config
from benchmark.evaluation.judge_package import RESULT_SCHEMA, validate, inspect, request_key, valid_result


def completion(config, key, system, payload):
    """DeepSeek OpenAI-compatible transport. Existing judge commands are unchanged."""
    body = dict(model=config['model'], temperature=0, max_tokens=config['max_tokens'],
                response_format={'type': 'json_object'},
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
    if config['thinking'] != 'provider-default':
        body['thinking'] = {'type': config['thinking']}
    request = urllib.request.Request(config['base_url'].rstrip('/') + '/chat/completions',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
        data=json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=config['timeout']) as response:
        result = json.load(response)
    choice = result['choices'][0]
    usage = dict(result.get('usage') or {})
    usage['_response_metadata'] = {k: result[k] for k in ('id', 'model', 'system_fingerprint', 'created') if k in result}
    usage['_response_metadata']['received_at'] = datetime.now(timezone.utc).isoformat()
    return choice['message'].get('content') or '', usage, choice.get('finish_reason') == 'stop'


def run_request(task, bundle, config, key):
    # Cache the raw, valid API answer. Per-task missing-call zeros are applied
    # later; those zeros must not make a raw null call score fail validation.
    row = dict(payload=task['row']['payload'], call_applicable=task['row']['payload']['judge_call'])
    result = evaluate_content(row, config, key, bundle['prompts'][task['rubric']]['text'], request=completion)
    valid_result(result, row['payload']['judge_call'])
    return result


def shape_task(task, result, key=None):
    row = task['row']
    status = result['status']
    response = result.get('response_score') if status == 'complete' else None
    call = result.get('call_score') if status == 'complete' else None
    if status == 'complete':
        if not row['payload']['candidate_response'].strip():response = 0
        if row['call_applicable'] and not row['payload']['judge_call']:call = 0
    task_score = None if status != 'complete' else (response+call)/2 if row['call_applicable'] else response
    return dict(task_id=task['task_id'], run_id=task['run_id'], qa_sha256=task['qa_sha256'], video=task['video'],
                **{k: row[k] for k in ('sample_index', 'response_index', 'mode', 'ability', 'call_applicable')},
                probe_id=row.get('probe_id'), matched=task['matched'], status=status,
                scoring_source='api' if task['api_required'] else 'rule', request_key=key,
                response_score=response, call_score=call, task_score=task_score,
                response_reason=result.get('response_reason'),
                call_reason=('Missing or malformed actual call' if status=='complete' and row['call_applicable']
                             and not row['payload']['judge_call'] else result.get('call_reason')),
                error_types=result.get('error_types', []))


def means(items):
    okay = [t for t in items if t['status'] == 'complete']
    complete = len(okay) == len(items)
    def mean(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values)/len(values) if values else None
    return dict(tasks=len(items), scored=len(okay), failed=sum(t['status']=='failed' for t in items),
                pending=sum(t['status']=='pending' for t in items),
                response_mean=mean(okay, 'response_score') if complete else None,
                call_mean=mean(okay, 'call_score') if complete else None,
                task_mean=mean(okay, 'task_score') if complete else None,
                scored_only_response_mean=mean(okay, 'response_score'), scored_only_task_mean=mean(okay, 'task_score'),
                response_mean_matched=(mean([t for t in okay if t['matched']], 'response_score')
                                       if all(t['status']=='complete' for t in items if t['matched']) else None))


def build_result(bundle, identity, cache, interrupted=False):
    tasks = []
    for task in bundle['tasks']:
        if task['api_required']:
            key = request_key(task, bundle)
            result = cache.get(key, {'status': 'pending'})
        else:
            key = None
            result = evaluate_content(task['row'], {'retries': 0}, '', '',
                                      matched=task['matched'], request=lambda *a: (_ for _ in ()).throw(AssertionError('Rule task called API')))
        tasks.append(shape_task(task, result, key))
    summary = {}
    for run in bundle['runs']:
        rows = [t for t in tasks if t['run_id'] == run['run_id']]
        groups = defaultdict(list)
        for row in rows:
            for name in ('overall', 'mode:'+row['mode'], 'ability:'+str(row['ability'])):
                groups[name].append(row)
        summary[run['run_id']] = {name: means(values) for name, values in groups.items()}
    totals = defaultdict(float)
    attempts = [a for r in cache.values() for a in r.get('usage_attempts', [])]
    for attempt in attempts:
        for k, v in attempt.get('usage', {}).items():
            if type(v) in (int, float):totals[k] += v
    return dict(schema=RESULT_SCHEMA, bundle_id=bundle['bundle_id'], identity=identity,
                generated_at=datetime.now(timezone.utc).isoformat(),
                status='interrupted' if interrupted else 'complete' if all(t['status']=='complete' for t in tasks) else 'incomplete',
                runs=bundle['runs'], tasks=tasks, summary=summary,
                requests=[dict(request_key=k, result=r) for k, r in sorted(cache.items())],
                api_usage=dict(attempts=len(attempts), failed_attempts=sum('error_type' in a for a in attempts),
                               usage_totals=dict(totals)),
                note='Scores 0..2. Agent task=(response+call)/2. Missing inference is coverage, not zero. '
                     'Probe scores are TP-conditional. Realtime includes rule zeros and matched-only means separately. '
                     'Authoritative means are null on incomplete groups; scored-only means are descriptive.')


def execute(args):
    if args.concurrency < 1 or args.timeout <= 0 or args.retries < 0 or args.max_tokens < 1:
        raise ValueError('Invalid concurrency, timeout, retries or token budget')
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input bundle and output must be different files')
    bundle = validate(json.loads(args.input.read_text()))
    description = inspect(bundle)
    print(json.dumps(description, indent=2))
    if args.dry_run:
        return
    if args.provider != 'openai' or not args.model:
        raise ValueError('Explicit --provider openai and --model are required for scoring')
    key, url = resolve_config(args)
    url_parts = urlparse(url)
    if url_parts.scheme not in ('https', 'http') or not url_parts.hostname or url_parts.username or url_parts.password or url_parts.query or url_parts.fragment:
        raise ValueError('Endpoint must not contain credentials, query parameters or fragments')
    config = dict(provider=args.provider, model=args.model, base_url=url, timeout=args.timeout,
                  retries=args.retries, max_tokens=args.max_tokens, thinking=args.thinking, temperature=0)
    identity = dict(schema=RESULT_SCHEMA, bundle_id=bundle['bundle_id'], **config)
    by_key = {}
    for task in bundle['tasks']:
        if task['api_required']:by_key.setdefault(request_key(task, bundle), task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = Path(str(args.output) + '.progress.jsonl')
    with Path(str(args.output) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = {}
        if args.output.exists():
            old = json.loads(args.output.read_text())
            if old.get('schema') != RESULT_SCHEMA or old.get('identity') != identity:
                raise ValueError('Output belongs to a different input or judge configuration')
            cache.update({r['request_key']: r['result'] for r in old['requests']})
        if history.exists():
            records = read_jsonl(history, complete_only=True)
            if not records or records[0] != {'kind': 'header', 'identity': identity}:
                raise ValueError('Resume journal identity changed')
            for record in records[1:]:cache[record['request_key']] = record['result']
            recover_jsonl_tail(history)
        else:
            append_jsonl(history, {'kind': 'header', 'identity': identity})
            for k, result in cache.items():append_jsonl(history, dict(request_key=k, result=result))
        for k, result in cache.items():
            if k not in by_key:raise ValueError('Unknown request key in returned results')
            valid_result(result, by_key[k]['row']['payload']['judge_call'])
        pending = [k for k in by_key if k not in cache or cache[k]['status']=='pending'
                   or (args.retry_failed and cache[k]['status']=='failed')]
        print(f'API requests pending: {len(pending)}; cached: {len(by_key)-len(pending)}')
        interrupted = False
        def accept(k, result):
            previous = cache.get(k, {}).get('usage_attempts', [])
            for index, attempt in enumerate(result['usage_attempts'], len(previous)+1):attempt['attempt'] = index
            result['usage_attempts'] = previous + result['usage_attempts']
            append_jsonl(history, dict(request_key=k, result=result))
            cache[k] = result
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_request, by_key[k], bundle, config, key): k for k in pending}
            received = set()
            try:
                for future in as_completed(futures):
                    accept(futures[future], future.result()); received.add(future)
                    print(f'{len(received)}/{len(pending)} API requests ({cache[futures[future]]["status"]})', flush=True)
            except KeyboardInterrupt:
                interrupted = True
                for future in futures:future.cancel()
                # Keep already-started, billable replies rather than requesting them again on resume.
                try:
                    for future in as_completed(futures):
                        if future not in received and not future.cancelled():
                            accept(futures[future], future.result()); received.add(future)
                finally:
                    write_json(args.output, build_result(bundle, identity, cache, interrupted=True))
            finally:
                write_json(args.output, build_result(bundle, identity, cache, interrupted))
        result = build_result(bundle, identity, cache, interrupted)
        write_json(args.output, result)
        print('Return this file: ' + str(args.output))
        if interrupted:raise SystemExit(130)
        if result['status'] != 'complete':raise SystemExit(2)
