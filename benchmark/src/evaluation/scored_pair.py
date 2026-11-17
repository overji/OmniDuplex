"""Content credit requires both correct Probe decisions; no F1 denominator."""
from collections import defaultdict
import math

from benchmark.evaluation.scored_f1 import normalized_quality

VERSION = 'scored_pair_v1'


def scored_pair_metrics(quality_sum, pair_count, correct_pair_count):
    for count in (pair_count, correct_pair_count):
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError('Scored-Pair counts must be nonnegative integers')
    result = dict(scored_pair_version=VERSION, quality_sum=quality_sum,
                  pair_count=pair_count, correct_pair_count=correct_pair_count, scored_pair=None)
    if pair_count is None or correct_pair_count is None:
        return result
    if correct_pair_count > pair_count:
        raise ValueError('Correct pairs exceed complete pairs')
    if quality_sum is None:
        return result
    if not math.isfinite(quality_sum) or not 0 <= quality_sum <= correct_pair_count + 1e-7:
        raise ValueError('Quality mass exceeds correct-pair bounds')
    result['scored_pair'] = quality_sum / pair_count if pair_count else None
    return result


def judged_pair_metrics(rows, population=None):
    """Only required correct-pair judgments can make the metric pending/null."""
    if population is None:
        return dict(**scored_pair_metrics(None, None, None),
                    scored_pair_unavailable_reason='Missing pair identities; re-export with original predictions')
    ids = set(population['correct_post_probe_ids'])
    if len(ids) != population['correct_pair_count']:
        raise ValueError('Correct-pair IDs must be unique and match the count')
    selected = {}
    for row in rows:
        pid = row.get('probe_id')
        if pid not in ids:
            continue
        if pid in selected:
            raise ValueError('Duplicate correct-pair judgment')
        selected[pid] = normalized_quality(row)
    quality = (sum(selected.values()) if set(selected) == ids
               and all(v is not None for v in selected.values()) else None)
    return scored_pair_metrics(quality, population['pair_count'], population['correct_pair_count'])


def probe_population(manifest, predictions):
    """Freeze complete-pair counts and post IDs whose pre/post decisions both pass."""
    from benchmark.probe.metrics import record_output
    index = {}
    for prediction in predictions:
        if prediction.get('status') != 'complete':
            continue
        pid = prediction['probe_id']
        if pid in index:
            raise ValueError('Duplicate probe prediction')
        index[pid] = prediction
    pairs = defaultdict(list)
    for row in manifest:
        pairs[row['pair_id']].append(row)
    groups = {}
    for rows in pairs.values():
        if len(rows) != 2 or not all(r['probe_id'] in index for r in rows):
            continue
        sides = {r['side']: r for r in rows}
        if set(sides) != {'pre', 'post'}:
            raise ValueError('A Probe pair must contain pre and post')
        pre, post = sides['pre'], sides['post']
        if any(pre[k] != post[k] for k in ('mode', 'ability')):
            raise ValueError('Probe pair groups differ')
        correct = (record_output(index[pre['probe_id']], pre)['decision'] == 'NO'
                   and record_output(index[post['probe_id']], post)['decision'] == 'YES')
        for name in ('overall', 'mode:' + post['mode'], 'ability:' + post['ability']):
            g = groups.setdefault(name, dict(pair_count=0, correct_pair_count=0, correct_post_probe_ids=[]))
            g['pair_count'] += 1
            g['correct_pair_count'] += int(correct)
            if correct:
                g['correct_post_probe_ids'].append(post['probe_id'])
    return groups
