"""Reply/window-local rewards. References are consulted only after rollout.

Units carry zero-baseline advantages and one normalization weight. Reply units
own all sampled tokens in an utterance; missed windows own idle decisions or,
when fully occupied, charge the complete blocking replies chosen by the user.
"""
import math
from collections import Counter

from benchmark.data.timestamps import seconds
from benchmark.evaluation.agent_calls import parse_event
from benchmark.evaluation.timing import match_windows, reference_windows
from train.src.domain.agent import training_sample
from train.src.domain.confusion_reward import confusion_reward


def local_reward(sample, trajectory, judgments, config):
    sample = training_sample(sample)
    # Reuse validation, Judge score semantics and official timing diagnostics.
    result = confusion_reward(sample, trajectory, judgments, config)
    chunks, events = [], trajectory['events']
    for chunk in trajectory['chunks']:
        if 'actions' in chunk and 'action_reply_ids' not in chunk:
            chunk = {**chunk, 'action_reply_ids': [a['reply_id'] for a in chunk['actions']]}
        chunks.append(chunk)
    refs = dict(reference_windows(sample))
    matched = match_windows(sample, events)
    matches = {m['event_index']: m['response_index'] for m in matched['matches']}
    weights, n = config['reward_weights'], len(refs)
    full_credit = config.get('reward_backend') == 'tp_full_credit'
    start = math.ceil(seconds(sample['question_time'])) + 1
    replies = {}
    for second, chunk in enumerate(chunks, 1):
        required = ('reply_ids', 'turn_open_before', 'action_reply_ids')
        if any(key not in chunk for key in required):
            raise ValueError('Local rewards require reply ownership telemetry; regenerate rollouts')
        owners = chunk['action_reply_ids']
        if type(chunk['turn_open_before']) is not bool:
            raise ValueError('Invalid native turn state')
        if any(owner is not None and (type(owner) is not int or owner < 0) for owner in owners):
            raise ValueError('Invalid reply action ownership')
        if second < start:
            continue
        for owner in owners:
            if owner is not None:
                replies.setdefault(owner, dict(kind='reply', reply_id=owner, advantage=0.,
                    base_advantage=0., blocked_fn_advantage=0., weight=1/n, action_count=0,
                    onset_seconds=second, event_index=None, response_index=None, blocked_windows=[]))['action_count'] += 1
    event_owners = set()
    tp_contribution = fp_contribution = invalid_contribution = 0.
    for index, event in enumerate(events):
        owner = event.get('reply_id')
        if owner not in replies or owner in event_owners:
            raise ValueError('Parsed reply does not map uniquely to native sampled actions')
        event_owners.add(owner)
        unit = replies[owner]
        unit['event_index'] = index
        if index in matches:
            ri = matches[index]
            answer = judgments[ri]
            parsed = parse_event(event)
            qr = 1. if full_credit else (answer['response_score']/2 if parsed['response'].strip() else 0.)
            q = qr
            if sample['mode'] == 'agent':
                from benchmark.data.agent_schema import validate_task_start
                valid = (parsed['status'] == 'structured' and len(parsed['calls']) == 1
                         and all(validate_task_start(call) for call in parsed['calls']))
                qa = 1. if full_credit else (answer['call_score']/2 if valid else 0.)
                q = (qr + qa)/2
            unit.update(response_index=ri, quality=q)
            unit['base_advantage'] = weights['tp']*(2*q-1)
            tp_contribution += unit['base_advantage']/n
        else:
            unit['base_advantage'] = -weights['fp']
            fp_contribution -= weights['fp']/n
        parsed = parse_event(event)
        from benchmark.data.agent_schema import validate_task_start
        valid = (parsed['status'] == 'structured' and len(parsed['calls']) == 1
                 and all(validate_task_start(call) for call in parsed['calls']))
        if parsed['attempted'] and not valid:
            unit['base_advantage'] -= weights['invalid_call_penalty']
            invalid_contribution -= weights['invalid_call_penalty']/n
    for owner, unit in replies.items():
        # Empty/control-only attempts are not allowed to vanish with text parsing.
        if owner not in event_owners:
            unit['base_advantage'] = -weights['fp']
            fp_contribution -= weights['fp']/n
    units = list(replies.values())
    hit_windows = set(matches.values())
    idle_fn = blocked_fn = unassigned_fn = 0
    unassigned_windows = []
    for ri, ref in refs.items():
        eligible = [s for s in range(start, len(chunks)+1)
                    if seconds(ref['window_start']) <= s <= seconds(ref['window_end'])]
        idle = [s for s in eligible if chunks[s-1]['idle_wait']
                and any(owner is None for owner in chunks[s-1]['action_reply_ids'])]
        if idle:
            units.append(dict(kind='fn', response_index=ri, seconds=idle,
                advantage=0. if ri in hit_windows else -weights['fn'], weight=1/n))
        if ri in hit_windows:
            continue
        if idle:
            idle_fn += 1
            continue
        occupied = Counter()
        for second in eligible:
            owners = [owner for owner in chunks[second-1]['reply_ids'] if owner in replies]
            for owner in owners:
                occupied[owner] += 1/len(owners)
        if occupied:
            total = sum(occupied.values())
            for owner, count in occupied.items():
                unit = replies[owner]
                unit['blocked_fn_advantage'] -= weights['fn']*count/total
                unit['blocked_windows'].append(dict(response_index=ri, fraction=count/total))
            blocked_fn += 1
        else:
            unassigned_fn += 1
            unassigned_windows.append(ri)
    windows = [(seconds(r['window_start']), seconds(r['window_end'])) for r in refs.values()]
    outside = [s for s in range(start, len(chunks)+1)
               if not chunks[s-1]['turn_open_before']
               and chunks[s-1]['action_reply_ids']
               and not any(a <= s <= b for a,b in windows)]
    waiting = [s for s in outside if chunks[s-1]['idle_wait']]
    if outside and waiting:
        # Each second is one decision, irrespective of its sampled token count.
        units.append(dict(kind='wait', seconds=waiting, advantage=1.,
                          weight=weights['tn']*len(waiting)/len(outside)))
    # Every ablation sees the same structural action coverage. These sampled
    # controls have no local reward, but share the trajectory advantage in the
    # group-relative/unit-weighted ablation.
    for second in range(start, len(chunks)+1):
        chunk = chunks[second-1]
        if not chunk['idle_wait'] and any(owner is None for owner in chunk['action_reply_ids']):
            units.append(dict(kind='control', seconds=[second], advantage=0., weight=1/n))
    for unit in replies.values():
        unit['advantage'] = unit['base_advantage'] + unit['blocked_fn_advantage']
    fn_contribution = -weights['fn']*matched['misses']/n
    tn_contribution = weights['tn']*len(waiting)/len(outside) if outside else 0.
    result.update(total=tp_contribution+fp_contribution+invalid_contribution+fn_contribution+tn_contribution,
        fp=len(replies)-matched['hits'], unparsed_response_starts=len(replies)-len(events),
        timing=2*matched['hits']/(n+len(replies)), silent=not replies,
        tn=len(waiting), negative_seconds=len(outside), silent_negative_seconds=len(waiting),
        silence=len(waiting)/len(outside) if outside else 0.,
        tp_contribution=tp_contribution, fp_contribution=fp_contribution,
        fn_contribution=fn_contribution, invalid_call_contribution=invalid_contribution,
        tn_contribution=tn_contribution,
        local_credit=dict(version=1, units=units, unassigned_windows=unassigned_windows,
                          start_second=start),
        local_idle_fn=idle_fn, local_blocked_fn=blocked_fn, local_unassigned_fn=unassigned_fn,
        local_reply_units=len(replies), local_wait_decisions=len(waiting),
        local_wait_opportunities=len(outside),
        local_positive_units=sum(u['advantage'] > 0 and u['weight'] > 0 for u in units),
        local_negative_units=sum(u['advantage'] < 0 and u['weight'] > 0 for u in units))
    return result
