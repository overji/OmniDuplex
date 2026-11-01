"""Event-normalized TP/FP/FN reward with bounded, clock-normalized waiting credit."""
import math
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import match_windows, reference_windows
from benchmark.evaluation.agent_calls import parse_event
from benchmark.data.agent_schema import validate_task_start
from benchmark.evaluation.judge import SCORES
from train.src.domain.agent import training_sample


def confusion_reward(sample, trajectory, judgments, config):
    sample = training_sample(sample)
    events, chunks = trajectory['events'], trajectory['chunks']
    windows = [(seconds(r['window_start']), seconds(r['window_end']))
               for _, r in reference_windows(sample)]
    if not windows or any(b > len(chunks) for _, b in windows):
        raise ValueError('Confusion reward requires complete reference-window coverage')
    start = math.ceil(seconds(sample['question_time'])) + 1
    negative = []
    starts = 0
    for second, chunk in enumerate(chunks, 1):
        if second < start:
            continue
        if type(chunk.get('new_response')) is not bool or type(chunk.get('idle_wait')) is not bool:
            raise ValueError('Confusion reward requires native new_response and idle_wait telemetry')
        if chunk['new_response'] and chunk['idle_wait']:
            raise ValueError('A fresh response cannot also be idle waiting')
        starts += chunk['new_response']
        if not any(a <= second <= b for a, b in windows):
            negative.append(chunk)
    matched = match_windows(sample, events)
    n, tp = matched['windows'], matched['hits']
    # Native control-only/empty attempts must not disappear with text parsing.
    unparsed = max(0, starts - len(events))
    fp, fn = len(events) - tp + unparsed, n - tp
    invalid = 0
    parsed = []
    for event in events:
        item = parse_event(event)
        valid_call = (item['status'] == 'structured' and len(item['calls']) == 1
                      and all(validate_task_start(c) for c in item['calls']))
        parsed.append((item, valid_call))
        invalid += bool(item['attempted'] and not valid_call)
    tp_sum = content = calls = 0.
    for match in matched['matches']:
        item = judgments.get(match['response_index'])
        if not item or item.get('status') != 'complete':
            raise ValueError('Failed judge must exclude the whole QA group')
        def score(key):
            value = item.get(key)
            if type(value) not in (int, float) or value not in SCORES:
                raise ValueError('Invalid judge ' + key)
            return value / 2
        response, valid_call = parsed[match['event_index']]
        qr = score('response_score')
        if not response['response'].strip():
            qr = 0.
        content += qr / n
        q = qr
        if sample['mode'] == 'agent':
            qa = score('call_score')
            if not valid_call:
                qa = 0.
            calls += qa / n
            q = (qr + qa) / 2
        tp_sum += 2 * q - 1
    tn = sum(c['idle_wait'] for c in negative)
    silence = 2 * tn / len(negative) - 1 if negative else 0.
    weights = config['reward_weights']
    terms = dict(tp=weights['tp'] * tp_sum / n,
                 fn=-weights['fn'] * fn / n,
                 fp=-weights['fp'] * fp / n,
                 tn=weights['tn'] * silence,
                 invalid_call=-weights['invalid_call_penalty'] * invalid / n)
    return dict(total=sum(terms.values()), timing=2*tp/(n+len(events)+unparsed),
                content=content, call=calls, silence=silence, hits=tp, tp=tp, fp=fp, fn=fn, tn=tn,
                reference_windows=n, negative_seconds=len(negative), silent_negative_seconds=tn,
                negative_response_starts=sum(c['new_response'] for c in negative),
                unparsed_response_starts=unparsed, invalid_calls=invalid,
                silent=not events and starts == 0,
                **{key+'_contribution': value for key,value in terms.items()})
