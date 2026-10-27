"""本地规则奖励：一对一时间匹配、文本相似度与调用匹配，不等价于语义judge。"""

import json
import unicodedata
from collections import Counter

from benchmark.evaluation.timing import match_windows
from benchmark.evaluation.agent_calls import parse_event


def text_f1(left, right):
    def grams(text):
        chars = ''.join(c for c in unicodedata.normalize('NFKC', str(text)).casefold()
                        if c.isalnum())
        return Counter(chars[i:i + 2] for i in range(len(chars) - 1)) if len(chars) > 1 else Counter(chars)
    a, b = grams(left), grams(right)
    if not a or not b:
        return float(not a and not b)
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))


def value_score(actual, expected):
    if type(actual) is not type(expected):
        return 0.
    if isinstance(expected, dict):
        keys = set(actual) | set(expected)
        return sum(value_score(actual[k], expected[k]) for k in keys if k in actual and k in expected) / max(1, len(keys)) if keys else 1.
    if isinstance(expected, list):
        return sum(value_score(a, b) for a, b in zip(actual, expected)) / max(len(actual), len(expected), 1) if actual or expected else 1.
    if isinstance(expected, str):
        return text_f1(actual, expected)
    return float(actual == expected)


def call_score(actual, expected):
    if len(actual) != len(expected) or not expected:
        return 0.
    result = 0.
    for a, b in zip(actual, expected):
        af, bf = a['function'], b['function']
        arguments = bf.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if af['name'] == bf['name']:
            result += value_score(af.get('arguments', {}), arguments)
    return result / len(expected)


def reward(sample, events, config):
    from train.src.domain.agent import training_sample
    from benchmark.data.agent_schema import validate_task_start
    sample = training_sample(sample)
    matched = match_windows(sample, events)
    n = max(matched['windows'], 1)
    timing = 2 * matched['hits'] / max(matched['windows'] + len(events), 1)
    content, calls = 0., 0.
    kind = sample.get('sample_type', sample.get('generation_context', {}).get('sample_type'))
    negative = kind in ('no_context', 'missing_tool')
    any_attempt = any(parse_event(event)['attempted'] for event in events)
    for match in matched['matches']:
        ref = sample['responses'][match['response_index']]
        event = parse_event(events[match['event_index']])
        # Empty output never earns reference-content or negative-call reward.
        if event['response'].strip():
            content += text_f1(event['response'], ref.get('response', ref.get('text', ref.get('content', '')))) / n
            if negative:
                calls += float(not any_attempt) / n
        if not negative and event['status'] == 'structured' and all(validate_task_start(c) for c in event['calls']):
            calls += call_score(event['calls'], ref.get('tool_calls', [])) / n
    invalid = sum(e.get('agent_parse_status') in ('malformed', 'truncated', 'duplicate_separator') for e in events)
    weights = config.get('reward_weights', {})
    if sample['mode'] == 'agent':
        total = weights.get('agent_timing', .4) * timing + weights.get('agent_content', .4) * content + weights.get('agent_call', .2) * calls
    else:
        total = weights.get('timing', .5) * timing + weights.get('content', .5) * content
    total -= weights.get('invalid_call_penalty', .1) * invalid
    return dict(total=total, timing=timing, content=content, call=calls,
                invalid_calls=invalid, hits=matched['hits'], silent=not events)
