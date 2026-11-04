"""Read-only, paired realtime summaries; never resumes jobs or changes scoring."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import time
from benchmark.data.files import read_jsonl,write_json,atomic_write_text
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import summarize,reference_windows


def load_predictions(directory):
    root=Path(directory);path=root/'predictions.jsonl'
    paths=[path] if path.exists() else sorted(root.glob('worker_*/predictions.jsonl'))
    result={}
    for path in paths:
        for row in read_jsonl(path,complete_only=True):
            index=row['sample_index']
            if index in result:raise ValueError('Duplicate prediction index')
            result[index]=row
    return result


def percent(value):
    return 'N/A' if value is None else f'{100*value:.2f}%'


def render(config):
    samples=json.loads(Path(config['data']).read_text());models={};judges={}
    for model in config['models']:
        name=model['name']
        if name in models:raise ValueError('Duplicate model label')
        if model.get('data') and json.loads(Path(model['data']).read_text())!=samples:
            raise ValueError('Model reference data differs; explicit sample alignment is required')
        rows=load_predictions(model['run'])
        if any(type(i) is not int or not 0<=i<len(samples) for i in rows):
            raise ValueError('Prediction index outside this dataset')
        models[name]=dict(all=rows,complete={i:r for i,r in rows.items() if r['status']=='complete'})
        if model.get('judge') and Path(model['judge']).exists():judges[name]=json.loads(Path(model['judge']).read_text())
    common=set.intersection(*(set(m['complete']) for m in models.values())) if models else set()
    mismatched=[]
    for i in sorted(common):
        rows=[m['complete'][i] for m in models.values()]
        durations=[r.get('observed_seconds') for r in rows]
        if (not all(r.get('system_prompt_sha256') for r in rows)
                or len({r['system_prompt_sha256'] for r in rows})!=1
                or not all(isinstance(d,(int,float)) and math.isfinite(d) for d in durations)
                or max(durations)-min(durations)>1e-5):mismatched.append(i)
    common-=set(mismatched)
    paired={name:summarize(samples,[m['complete'][i] for i in sorted(common)]) for name,m in models.items()}
    now=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    lines=[f"# {config['title']}",'',f'Updated: {now}. Refresh interval: {config.get("interval_seconds",60)} s.','',
           'Timing-only inclusive windows, one-to-one response onsets. Content is judged separately. Partial progress is not a final balanced estimate.','',
           '## Progress — each model’s own completed subset','',
           '| Model | Complete / total | Errors | Recall | Precision | F1 |','|---|---:|---:|---:|---:|---:|']
    for note in config.get('notes', []):
        lines.insert(4, str(note)+'\n')
    for name,m in models.items():
        own=summarize(samples,list(m['complete'].values()))['groups'].get('overall',{})
        errors=len(m['all'])-len(m['complete'])
        lines.append(f"| {name} | {len(m['complete'])}/{len(samples)} | {errors} | {percent(own.get('window_hit_rate'))} | {percent(own.get('timing_precision'))} | {percent(own.get('timing_f1'))} |")
    lines+=['','## Paired comparison', '',f'Identical completed QA intersection: **{len(common)}**. Prompt/duration mismatches excluded: {len(mismatched)}. Reference data must match. Historical decoding settings are not inferred from predictions; compare compatible runs.','',
            '| Model | TP | FP | FN | Recall | Precision | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,metric in paired.items():
        g=metric['groups'].get('overall',{})
        if not g:lines.append(f'| {name} | N/A | N/A | N/A | N/A | N/A | N/A |');continue
        lines.append(f"| {name} | {g['hits']} | {g['predictions']-g['hits']} | {g['windows']-g['hits']} | {percent(g['window_hit_rate'])} | {percent(g['timing_precision'])} | {percent(g['timing_f1'])} |")
    first_later={};breakdowns={}
    for dimension in ('mode','task','length'):
        buckets=defaultdict(list)
        for i in sorted(common):
            if dimension=='length':
                d=samples[i].get('duration_seconds')
                if not isinstance(d,(int,float)):d=next(iter(models.values()))['complete'][i]['observed_seconds']
                key='≤60s' if d<=60 else '60–180s' if d<=180 else '180–300s' if d<=300 else '>300s'
            else:key=samples[i][dimension]
            buckets[key].append(i)
        lines+=['',f'## {dimension} F1 — same paired QA','', '| Group | QA | '+' | '.join(models)+' |', '|---|---:|'+ '|'.join('---:' for _ in models)+'|']
        for key,indices in sorted(buckets.items()):
            scores={name:summarize(samples,[m['complete'][i] for i in indices])['groups']['overall'] for name,m in models.items()}
            breakdowns[dimension+':'+key]=scores
            values=[g['timing_f1'] for g in scores.values()]
            best=max(values)
            lines.append(f'| {key} | {len(indices)} | '+' | '.join(('**'+percent(v)+'**') if v==best else percent(v) for v in values)+' |')
    lines+=['','## First / later window recall','', '| Model | First | Later |','|---|---:|---:|']
    for name,metric in paired.items():
        totals=Counter()
        for row in metric['per_sample']:
            refs=sorted(reference_windows(samples[row['sample_index']]),key=lambda pair:(seconds(pair[1]['window_start']),pair[0]))
            if not refs:continue
            hits={m['response_index'] for m in row['matches']}
            totals.update(first_n=1,first_tp=int(refs[0][0] in hits),later_n=len(refs)-1,later_tp=sum(ri in hits for ri,_ in refs[1:]))
        first_later[name]=dict(totals)
        lines.append(f"| {name} | {percent(totals['first_tp']/totals['first_n'] if totals['first_n'] else None)} | {percent(totals['later_tp']/totals['later_n'] if totals['later_n'] else None)} |")
    if judges:
        lines+=['','## Content judge — own scored windows, scale 0–2','', '| Model | Status | Scored / total | Failed | Matched mean | All-window mean |','|---|---|---:|---:|---:|---:|']
        for name,j in judges.items():
            g=j.get('groups',{}).get('overall',{});fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
            lines.append(f"| {name} | {j.get('status','pending')} | {g.get('scored_windows',0)}/{g.get('windows',0)} | {g.get('failed_windows',0)} | {fmt(g.get('response_mean_matched'))} | {fmt(g.get('response_mean_all_windows'))} |")
        lines+=['','Content means are not paired across models. Failed judge rows remain failures, not zero scores.']
    output=Path(config['output']);output.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_text(output,'\n'.join(lines)+'\n')
    write_json(output.with_suffix('.json'),dict(updated_at=now,paired_qa=len(common),excluded_mismatches=mismatched,
               paired=paired,first_later=first_later,breakdowns=breakdowns,progress={n:dict(complete=len(m['complete']),terminal=len(m['all']),total=len(samples)) for n,m in models.items()}))
    return len(common)


def change(value, baseline):
    """Signed percentage-point change; missing observations must remain N/A."""
    if value is None or baseline is None:
        return 'N/A'
    delta=round(100*(value-baseline),2)
    if delta==0:return '→ 0.00 pp'
    return f'{"↑" if delta>0 else "↓"} {delta:+.2f} pp'


def comparison_section(report, data):
    """A self-contained index section with scores and deltas on identical QA."""
    names=[m['name'] for m in report['models']]
    candidate=report.get('candidate',names[-1])
    if candidate not in names:raise ValueError('Unknown candidate model')
    baselines=[name for name in names if name!=candidate]
    lines=[f"## {report['title']}",'',
           ' · '.join(f"{name}: {data['progress'][name]['complete']}/{data['progress'][name]['total']}" for name in names),'',
           f"共同成功 QA：**{data['paired_qa']}**；排除 prompt/时长不一致：{len(data['excluded_mismatches'])}。以下分数与升降均基于同一批 QA。",'',
           f"变化方向：**{candidate} 相对各历史模型**；单位 pp 为百分点。未完成子集不是最终成绩。",'']
    lines.extend(str(note)+'\n' for note in report.get('notes',[]))
    header='| '+' | '.join(['指标',*names,*('Δ vs '+name for name in baselines)])+' |'
    separator='|---|'+'|'.join('---:' for _ in range(len(names)+len(baselines)))+'|'
    lines += [header,separator]
    def append(label,values):
        valid=[v for v in values.values() if v is not None];best=max(valid) if valid else None
        cells=[('**'+percent(values[n])+'**') if values[n] is not None and values[n]==best else percent(values[n]) for n in names]
        cells += [change(values[candidate],values[n]) for n in baselines]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    for label,key in [('总 Recall','window_hit_rate'),('总 Precision','timing_precision'),('总 F1','timing_f1')]:
        append(label,{n:data['paired'][n]['groups'].get('overall',{}).get(key) for n in names})
    for label,hit,total in [('首次窗口 Recall','first_tp','first_n'),('后续窗口 Recall','later_tp','later_n')]:
        append(label,{n:(data['first_later'][n].get(hit,0)/data['first_later'][n][total]
                         if data['first_later'][n].get(total) else None) for n in names})
    if data.get('breakdowns'):
        lines+=['','各分组 F1（仍使用上述共同完成子集；行内最优值加粗）：','',header,separator]
        for key,scores in data['breakdowns'].items():
            count=next(iter(scores.values()))['samples']
            append(f'{key} · n={count}',{n:scores[n]['timing_f1'] for n in names})
    lines+=['',f"[详细结果、TP/FP/FN 与内容 Judge]({report['output']})",'']
    return lines
