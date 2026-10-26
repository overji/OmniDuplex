"""Offline content judge for true-positive reply/silent probes only."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
from io import StringIO
import json
from pathlib import Path

from benchmark.data.agent_schema import adapt_sample, validate_task_start
from benchmark.data.files import append_jsonl, atomic_write_text, read_jsonl, recover_jsonl_tail, write_json
from benchmark.data.timestamps import seconds
from benchmark.evaluation import judge
from benchmark.evaluation.judge_client import completion, resolve_config
from benchmark.evaluation.timing import reference_windows
from benchmark.probe.data import SCHEMA, digest, file_hash, load_manifest
from benchmark.probe.output import PROTOCOL, parse_reply

VERSION = 'probe_tp_content_v1'
PROBE_RUBRIC = '''
This is an independent prefix probe, not an online timing match. Only true-positive
reply decisions are supplied. Score the actual user-facing response against the
current reference, not an explanation that a response is warranted. Repeated
prompts, timestamp loops and reasoning tags are not substitutes for a response.
Score only the emitted text; never repair truncation or invent missing content.
For Agent evaluate task_start(instruction), without demanding legacy JSON fields.
No tool has been executed. Do not assume later tool results or completion replies.
Response and call scores are independent. Call validity/missing-call zeros and
the final arithmetic average are applied by code, not by you.
'''


def validate_manifest(samples, manifest, data_hash, videos_dir=None):
    """Bind sample/response IDs, reference windows and pair IDs to the source JSON."""
    cutoffs = {row['pair_id']:row['prefix_end_seconds'] for row in manifest if row['side']=='pre'}
    for row in manifest:
        si, ri = row['sample_index'], row['response_index']
        if type(si) is not int or not 0 <= si < len(samples):
            raise ValueError('Manifest sample_index is invalid')
        sample = samples[si]
        if type(ri) is not int or not 0 <= ri < len(sample['responses']):
            raise ValueError('Manifest response_index is invalid')
        reference = sample['responses'][ri]
        if ri not in dict(reference_windows(sample)):
            raise ValueError('Manifest points at a non-scored response')
        if any(row[k] != sample[k] for k in ('mode', 'task', 'ability')):
            raise ValueError('Manifest task identity changed')
        if any(row['request'][k] != sample[k] for k in ('question', 'question_time')):
            raise ValueError('Manifest request changed')
        if (row['request'].get('available_tools', []) != sample.get('available_tools', [])
                or row['window_start_seconds'] != seconds(reference['window_start'])
                or row['window_end_seconds'] != seconds(reference['window_end'])):
            raise ValueError('Manifest source/reference identity changed')
        if videos_dir is not None and (videos_dir/sample['video']).resolve() != Path(row['video']).resolve():
            raise ValueError('Manifest video alias identity changed')
        expected_pair = digest([SCHEMA, data_hash, si, ri, cutoffs[row['pair_id']], row['window_end_seconds']])
        if row['pair_id'] != expected_pair or row['probe_id'] != expected_pair + ':' + row['side']:
            raise ValueError('Manifest data fingerprint/pair identity changed')


def prepare_rows(samples, manifest, predictions):
    lookup = {row['probe_id']: row for row in manifest}
    seen, index, counts, rows, adapted = set(), {}, Counter(), [], {}
    for prediction in predictions:
        pid = prediction['probe_id']
        if pid not in lookup or pid in seen or prediction.get('status') != 'complete':
            raise ValueError('Unknown, duplicate or incomplete probe prediction')
        seen.add(pid)
        if prediction.get('output_protocol') != PROTOCOL:
            raise ValueError('Probe judge supports only reply-task-text; old/mixed protocols require separate decision scoring')
        source = lookup[pid]
        if any(prediction.get(k) != source[k] for k in ('sample_index', 'response_index', 'side', 'pair_id')):
            raise ValueError('Prediction/manifest identity mismatch')
        raw, complete = prediction.get('raw_response'), prediction.get('task_text_generation_complete')
        if not isinstance(raw, str) or (complete is not None and type(complete) is not bool):
            raise ValueError('Invalid raw probe output or completion flag')
        index[pid] = judge.fingerprint(prediction)
        parsed = parse_reply(raw, source['mode'], complete=complete)
        decision = parsed['decision']
        classification = ('invalid' if decision is None else
            'TP' if source['expected'] == 'YES' and decision == 'YES' else
            'FN' if source['expected'] == 'YES' else 'FP' if decision == 'YES' else 'TN')
        counts[classification] += 1
        if classification != 'TP':
            continue
        si, ri = source['sample_index'], source['response_index']
        if si not in adapted:
            adapted[si] = adapt_sample(samples[si], 'task-start')
        sample = adapted[si]
        reference = sample['responses'][ri]
        ref = {k: reference[k] for k in ('response', 'tool_calls', 'evidence', 'details', 'trigger') if k in reference}
        if 'evidence' in ref:
            ref['evidence'] = [e for e in ref['evidence'] if ('available_time' in e or 'time' in e)
                and seconds(e.get('available_time', e.get('time'))) <= source['prefix_end_seconds']]
        agent = sample['mode'] == 'agent'
        valid_call = (agent and bool(sample.get('available_tools')) and bool(parsed['tool_calls'])
                      and all(validate_task_start(call) for call in parsed['tool_calls']))
        payload = dict(question=sample['question'], question_time=sample['question_time'],
            mode=sample['mode'], ability=sample['ability'], sample_type=sample.get('sample_type'),
            prefix_end_seconds=source['prefix_end_seconds'], reference=ref,
            candidate_response=parsed['response_text'],
            candidate_tool_calls=parsed['tool_calls'] if valid_call else [], judge_call=bool(valid_call))
        if agent:
            payload.update(available_tools=sample.get('available_tools', []),
                frontend_context={k:v for k,v in (sample.get('frontend_context') or {}).items()
                                  if k in ('system_prompt', 'tools')})
        rows.append(dict(probe_id=pid, sample_index=si, response_index=ri,
            mode=sample['mode'], ability=sample['ability'], call_applicable=agent,
            parse_status=parsed['agent_parse_status'], output_format_valid=parsed['output_format_valid'],
            generation_complete=complete, input_sha256=index[pid], payload=payload))
    return rows, index, dict(counts)


def evaluate(row, config, key, system, request=completion):
    result = judge.evaluate_content(row, config, key, system, request)
    result['task_score'] = (None if result['status'] != 'complete' else
        (result['response_score'] + result['call_score']) / 2 if row['call_applicable']
        else result['response_score'])
    return result


def summarize(rows, counts, inference_count, manifest_count):
    groups = defaultdict(list, {'overall': [], 'mode:rtd': [], 'mode:pr': [], 'mode:agent': []})
    for row in rows:
        for name in ('overall', 'mode:'+row['mode'], 'ability:'+str(row['ability'])):
            groups[name].append(row)
    result = {}
    def mean(values):
        return sum(values)/len(values) if values else None
    for name, items in groups.items():
        ok = [r for r in items if r['status'] == 'complete']
        agents = [r for r in items if r['call_applicable']]
        agents_ok = [r for r in ok if r['call_applicable']]
        result[name] = dict(tp_probes=len(items), scored_tp=len(ok),
            failed_tp=sum(r['status']=='failed' for r in items),
            pending_tp=sum(r['status']=='pending' for r in items),
            judge_coverage=len(ok)/len(items) if items else None,
            task_mean_tp=mean([r['task_score'] for r in ok]) if len(ok)==len(items) else None,
            response_mean_tp=mean([r['response_score'] for r in ok]) if len(ok)==len(items) else None,
            call_mean_tp=mean([r['call_score'] for r in agents_ok]) if len(agents_ok)==len(agents) else None,
            scored_task_mean=mean([r['task_score'] for r in ok]),
            scored_response_mean=mean([r['response_score'] for r in ok]),
            scored_call_mean=mean([r['call_score'] for r in agents_ok]))
    return dict(evaluation=VERSION, status='complete' if all(r['status']=='complete' for r in rows) else 'incomplete',
        scope='TP only; completion and means apply to this input snapshot, not the full inference run',
        manifest_probes=manifest_count, completed_inference_probes=inference_count,
        decisions=counts, groups=result,
        note='TP-conditional content scores, not overall benchmark performance. Agent task score=(response+call)/2; missing/invalid calls score zero. No video or future tool results sent.')


def write_report(path, metrics):
    fmt = lambda value: 'N/A' if value is None else f'{value:.3f}'
    lines = ['# Probe TP content judge', '', 'Status: '+metrics['status'], '', metrics['scope'], '', metrics['note'], '',
        f"Inference snapshot: {metrics['completed_inference_probes']}/{metrics['manifest_probes']} probes; decisions: {metrics['decisions']}", '',
        '| Group | Scored / TP | Response mean | Call mean | Task mean | Scored-only task mean |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, group in metrics['groups'].items():
        lines.append(f"| {name} | {group['scored_tp']}/{group['tp_probes']} | " +
            ' | '.join(fmt(group[k]) for k in ('response_mean_tp','call_mean_tp','task_mean_tp','scored_task_mean'))+' |')
    atomic_write_text(path, '\n'.join(lines)+'\n')


def execute(args):
    if args.concurrency < 1 or args.timeout <= 0 or args.retries < 0:
        raise ValueError('Invalid concurrency, timeout or retries')
    if len({p.resolve() for p in args.runs}) != len(args.runs) or len({p.name for p in args.runs}) != len(args.runs):
        raise ValueError('Run directories must be unique and have distinct names')
    samples = json.loads(args.data.read_text()); manifest = load_manifest(args.manifest)
    manifest_config = args.manifest.with_name('probe_manifest_config.json')
    videos_dir = None
    if manifest_config.is_file():
        source = read_json(manifest_config)
        if (source['data_sha256'] != file_hash(args.data)
                or source['manifest_sha256'] != file_hash(args.manifest)):
            raise ValueError('Manifest config source fingerprint changed')
        videos_dir = Path(source['videos_dir'])
    validate_manifest(samples, manifest, file_hash(args.data), videos_dir)
    prepared = []
    for run in args.runs:
        path = run/'probe_predictions.jsonl'
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, index, counts = prepare_rows(samples, manifest, read_jsonl(path, complete_only=True))
        prepared.append((run, rows, index, counts))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, runs=[dict(run=str(run), completed_probes=len(index),
            decisions=counts, tp_probes=len(rows), expected_api_requests=sum(bool(r['payload']['candidate_response'].strip()
            or r['payload']['judge_call']) for r in rows)) for run,rows,index,counts in prepared]), indent=2))
        return
    if not args.provider or not args.model:
        raise ValueError('--provider and --model are required except for --dry-run')
    key, url = resolve_config(args)
    system = args.system_prompt.read_text().strip()+'\n\n'+PROBE_RUBRIC.strip()
    implementation = judge.fingerprint({str(p):file_hash(p) for p in
        (Path(__file__), Path(judge.__file__), Path(__file__).with_name('judge_client.py'),
         Path(__file__).parents[1]/'probe/output.py', Path(__file__).parents[1]/'data/task_text.py')})
    config = dict(version=VERSION, output_protocol=PROTOCOL, provider=args.provider, model=args.model,
        base_url=url, timeout=args.timeout, retries=args.retries, temperature=0,
        system_prompt_sha256=judge.fingerprint(system), data_sha256=file_hash(args.data),
        manifest_sha256=file_hash(args.manifest), implementation_sha256=implementation)
    comparisons = []
    for run, rows, index, counts in prepared:
        out = args.output_dir/run.name; out.mkdir(parents=True, exist_ok=True)
        with (out/'.judge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity = {**config, 'run': str(run.resolve())}
            config_path = out/'judge_config.json'; history_path = out/'judgments.jsonl'; index_path = out/'input_index.json'
            if config_path.exists() and read_json(config_path) != identity:
                raise ValueError('Judge inputs/config changed; choose a new output directory')
            if not config_path.exists() and (history_path.exists() or index_path.exists()):
                raise ValueError('Existing probe judgments have no config')
            previous = read_json(index_path) if index_path.exists() else {}
            if any(pid not in index or index[pid]!=value for pid,value in previous.items()):
                raise ValueError('Existing probe predictions changed or disappeared; choose a new output directory')
            write_json(config_path, identity); write_json(index_path, index)
            recover_jsonl_tail(history_path)
            history = read_jsonl(history_path)
            cache = {r['cache_key']:r for r in history if r['status']=='complete'}
            results, pending = [], []
            for row in rows:
                row['cache_key'] = judge.fingerprint(dict(config=identity, row=row))
                if row['cache_key'] in cache:
                    results.append(cache[row['cache_key']])
                else:
                    pending.append(row)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(evaluate, row, config, key, system) for row in pending]
                for future in as_completed(futures):
                    result = future.result(); append_jsonl(history_path,result)
                    history.append(result); results.append(result)
                    done = {r['probe_id'] for r in results}
                    metrics = summarize(results+[dict(r,status='pending') for r in pending
                        if r['probe_id'] not in done],counts,len(index),len(manifest))
                    write_json(out/'judge_metrics.json',metrics)
                    print(f"{run.name}: {len(results)}/{len(rows)} ({result['status']})",flush=True)
            results.sort(key=lambda r:(r['sample_index'],r['response_index']))
            metrics = summarize(results,counts,len(index),len(manifest))
            write_json(out/'judge_metrics.json',metrics); write_report(out/'judge_report.md',metrics)
            attempts = [attempt for r in history for attempt in r['usage_attempts']]
            tokens = defaultdict(float)
            for attempt in attempts:
                for name,value in attempt['usage'].items():
                    if type(value) in (int,float):
                        tokens[name] += value
            write_json(out/'api_usage.json',dict(requests=len(attempts),failed_attempts=sum('error_type' in a for a in attempts),usage_totals=dict(tokens)))
            comparisons.append(dict(model=run.name,status=metrics['status'],**metrics['groups']['overall']))
    write_json(args.output_dir/'judge_comparison.json',comparisons)
    stream = StringIO(); fields = list(dict.fromkeys(k for row in comparisons for k in row))
    writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(comparisons)
    atomic_write_text(args.output_dir/'judge_comparison.csv',stream.getvalue())
    if any(row['status']!='complete' for row in comparisons):
        raise SystemExit(1)


def read_json(path):
    return json.loads(path.read_text())
