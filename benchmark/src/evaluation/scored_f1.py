"""Content-weighted one-to-one F1, independent of model/transport dependencies."""
import math
from collections import defaultdict

VERSION = 'scored_f1_v1'


def normalized_quality(row):
    """Normalize each completed 0..2 judgment; never turn failed requests into zero."""
    if row.get('status') != 'complete':
        return None
    response = row['response_score']
    call = row.get('call_score')
    values = [response, call] if row['call_applicable'] else [response]
    if any(type(v) not in (int,float) or v not in (0,.5,1,1.5,2) for v in values):
        raise ValueError('Invalid content judgment')
    return sum(values)/(2*len(values))


def scored_metrics(quality_sum, reference_count, prediction_count):
    """Fractions in [0,1]. An unknown denominator or failed Judge yields null."""
    for count in (reference_count,prediction_count):
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError('Scored-F1 counts must be nonnegative integers')
    result=dict(scored_f1_version=VERSION,quality_sum=quality_sum,
                reference_count=reference_count,prediction_count=prediction_count,
                scored_precision=None,scored_recall=None,scored_f1=None)
    if quality_sum is None or reference_count is None or prediction_count is None:
        return result
    if (not math.isfinite(quality_sum) or quality_sum < 0
            or quality_sum > min(reference_count,prediction_count)+1e-7):
        raise ValueError('Quality mass exceeds one-to-one matching bounds')
    result.update(scored_precision=quality_sum/prediction_count if prediction_count else 0.,
                  scored_recall=quality_sum/reference_count if reference_count else None,
                  scored_f1=2*quality_sum/(reference_count+prediction_count) if reference_count+prediction_count else None)
    return result


def judged_metrics(rows, reference_count, prediction_count):
    values=[normalized_quality(r) for r in rows]
    quality=None if any(v is None for v in values) else sum(values)
    return scored_metrics(quality,reference_count,prediction_count)


def realtime_population(samples,predictions):
    from benchmark.evaluation.timing import reference_windows
    groups={}
    for prediction in predictions:
        sample=samples[prediction['sample_index']]
        for name in ('overall','mode:'+sample['mode'],'ability:'+str(sample.get('ability'))):
            g=groups.setdefault(name,dict(reference_count=0,prediction_count=0))
            g['reference_count']+=len(reference_windows(sample))
            g['prediction_count']+=len(prediction['events'])
    return groups


def probe_population(manifest,predictions):
    """Use complete pairs; count YES on either side and invalid outputs as attempts.

    This lets pre-side false responses reduce scored-F1. Incomplete inference
    pairs are coverage, not zeros. Invalid outputs get no quality credit.
    """
    from benchmark.probe.metrics import record_output
    index={p['probe_id']:p for p in predictions if p.get('status')=='complete'}
    pairs=defaultdict(list)
    for row in manifest:pairs[row['pair_id']].append(row)
    groups={}
    for rows in pairs.values():
        if len(rows)!=2 or not all(r['probe_id'] in index for r in rows):continue
        first=rows[0]
        for name in ('overall','mode:'+first['mode'],'ability:'+first['ability']):
            group=groups.setdefault(name,dict(reference_count=0,prediction_count=0,tp_probe_ids=[]))
            group['reference_count']+=1
            for row in rows:
                decision=record_output(index[row['probe_id']],row)['decision']
                group['prediction_count']+=int(decision in ('YES',None))
                if row['side']=='post' and decision=='YES':group['tp_probe_ids'].append(row['probe_id'])
    return groups
