"""Complete-axis classification metrics and official window matching."""
from benchmark.evaluation.timing import summarize


def binary_metrics(labels, probabilities, threshold):
    rows=[(int(y),float(p)) for y,p in zip(labels,probabilities) if y in (0,1)]
    tp=sum(y==1 and p>=threshold for y,p in rows);fp=sum(y==0 and p>=threshold for y,p in rows)
    fn=sum(y==1 and p<threshold for y,p in rows);tn=sum(y==0 and p<threshold for y,p in rows)
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    groups={}
    for y,p in rows:groups.setdefault(p,[]).append(y)
    positives=tp+fn;negatives=fp+tn;below=0;rank_sum=0.;ap=0.;seen=0;seen_positive=0
    for p,ys in sorted(groups.items()):
        npos=sum(ys);nneg=len(ys)-npos
        rank_sum+=npos*(below+.5*nneg);below+=nneg
    for p,ys in sorted(groups.items(),reverse=True):
        npos=sum(ys);seen+=len(ys);seen_positive+=npos
        if positives:ap+=npos/positives*seen_positive/seen
    return dict(labelled_seconds=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        false_positive_rate=fp/negatives if negatives else None,
        auroc=rank_sum/(positives*negatives) if positives and negatives else None,
        auprc=ap if positives else None)


def evaluate(rows, predictions, threshold):
    by_index={p['sample_index']:p for p in predictions};labels=[];probabilities=[];completed=[];samples=[]
    delays=[];duplicate=0
    from benchmark.evaluation.timing import match_windows
    for row in rows:
        p=by_index.get(row['sample_index'])
        if not p or p['status']!='complete':continue
        labels+=row['labels'];probabilities+=p['probabilities']
        mapped={**p,'sample_index':len(samples)};samples.append(row['sample']);completed.append(mapped)
        matches=match_windows(row['sample'],p['events'])
        delays += [m['onset_seconds']-m['window_start_seconds'] for m in matches['matches']]
        duplicate += sum(any(m['window_start_seconds']<=e['onset_seconds']<=m['window_end_seconds'] for m in matches['matches'])
                         and i not in {m['event_index'] for m in matches['matches']} for i,e in enumerate(p['events']))
    timing=summarize(samples,completed)
    overall=timing['groups'].get('overall',{})
    observed=sum(len(p['probabilities']) for p in completed)
    traces=[t for p in completed for t in p.get('trace',[])]
    mean=lambda key: sum(t[key] for t in traces if key in t)/max(1,sum(key in t for t in traces))
    return dict(classification=binary_metrics(labels,probabilities,threshold),timing=timing,
        duplicate_triggers=duplicate,mean_trigger_delay_seconds=sum(delays)/len(delays) if delays else None,
        missed_windows=overall.get('windows',0)-overall.get('hits',0),
        extra_triggers_per_minute=(overall.get('predictions',0)-overall.get('hits',0))*60/observed if observed else None,
        mean_gate_seconds=mean('gate_seconds'),mean_prefill_seconds=mean('backend_prefill_seconds'),
        mean_generation_seconds=mean('generation_seconds'),max_queue_depth=max([t['queue_depth'] for t in traces] or [0]),
        incomplete_replies=sum(not e.get('complete',True) for p in completed for e in p['events']),
        completed_qa=len(completed),expected_qa=len(rows),failed_or_missing_qa=len(rows)-len(completed))
