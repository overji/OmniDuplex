"""Portable content-judge identities and validation, with no model/media imports."""
import math

from benchmark.evaluation.judge import fingerprint

SCHEMA = 'omniduplex-judge-input-v1'
RESULT_SCHEMA = 'omniduplex-judge-results-v1'


def content_hash(bundle):
    return fingerprint({k: v for k, v in bundle.items() if k not in ('bundle_id', 'created_at')})


def task_id(task):
    return fingerprint({k: v for k, v in task.items() if k != 'task_id'})


def request_key(task, bundle):
    return fingerprint({'rubric': bundle['prompts'][task['rubric']]['sha256'], 'payload': task['row']['payload']})


def needs_api(row):
    return row.get('event_index', 0) is not None and bool(row['payload']['candidate_response'].strip() or row['payload']['judge_call'])


def validate(bundle):
    if bundle.get('schema') != SCHEMA or bundle.get('bundle_id') != content_hash(bundle):
        raise ValueError('Unsupported or changed judge input bundle')
    runs = {r['run_id']: r for r in bundle['runs']}
    if len(runs) != len(bundle['runs']):
        raise ValueError('Duplicate run ID')
    for prompt in bundle['prompts'].values():
        if fingerprint(prompt['text']) != prompt['sha256']:
            raise ValueError('Rubric hash mismatch')
    seen = set()
    for task in bundle['tasks']:
        tid = task['task_id']
        if tid in seen or tid != task_id(task):
            raise ValueError('Duplicate or changed task ID')
        seen.add(tid)
        if task['run_id'] not in runs or task['rubric'] not in bundle['prompts']:
            raise ValueError('Unknown run/rubric')
        row = task['row']; payload = row['payload']
        if type(row['sample_index']) is not int or type(row['response_index']) is not int:
            raise ValueError('Invalid source indices')
        if not isinstance(payload['candidate_response'], str) or type(payload['judge_call']) is not bool:
            raise ValueError('Invalid candidate output')
        if type(row['call_applicable']) is not bool or type(task['api_required']) is not bool or task['api_required'] != needs_api(row):
            raise ValueError('Changed deterministic scoring rule')
        if task['matched'] != (row.get('event_index', 0) is not None):
            raise ValueError('Changed match identity')
    return bundle


def inspect(bundle):
    required = [t for t in bundle['tasks'] if t['api_required']]
    return dict(bundle_id=bundle['bundle_id'], runs=len(bundle['runs']), tasks=len(bundle['tasks']),
                api_tasks=len(required), rule_tasks=len(bundle['tasks'])-len(required),
                unique_api_requests=len({request_key(t, bundle) for t in required}))


def valid_result(result, judge_call):
    from benchmark.evaluation.judge import validate_answer
    import json
    if result.get('status') not in ('complete', 'failed', 'pending'):
        raise ValueError('Invalid request status')
    if result['status'] == 'complete':
        validate_answer(json.dumps(result), judge_call)
    for attempt in result.get('usage_attempts', []):
        if type(attempt.get('attempt')) is not int or attempt['attempt'] < 1:
            raise ValueError('Invalid attempt history')
        if not isinstance(attempt.get('usage'), dict):
            raise ValueError('Invalid API usage')
        for value in attempt['usage'].values():
            if type(value) in (int, float) and not math.isfinite(value):
                raise ValueError('Nonfinite API usage')
