"""AOI timing/content reporting for OmniPro; keeps incomplete inference separate."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools
import threading
from pathlib import Path
import urllib.request
from benchmark.data.files import read_jsonl, write_json, append_jsonl, atomic_write_text, recover_jsonl_tail
from benchmark.evaluation.timing import summarize, reference_windows
from benchmark.evaluation.judge import prepare_rows, evaluate, fingerprint, summarize as content_summary


def predictions(directory):
    rows = read_jsonl(Path(directory)/'predictions.jsonl', complete_only=True)
    indices = [r['sample_index'] for r in rows]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate prediction sample index')
    return rows


def score_run(config, label):
    """Same AOI payload/rubric/validation; local Qwen transport disables thinking."""
    root=Path(config['output']); run=root/'runs'/label; out=root/'judge'/label
    samples=json.loads(Path(config['data']).read_text()); all_predictions=predictions(run)
    rows, negatives=prepare_rows(samples, all_predictions)
    settings=config['judge']; system=Path(settings['system_prompt']).read_text()
    from train.src.infrastructure.judge import identity
    judge_identity=identity(settings)
    transport=dict(provider='openai', model=settings['model'], base_url=settings['base_url'],
                   timeout=settings['timeout'], retries=settings['retries'])
    binding=dict(version='omnipro_aoi_content_v1', judge=judge_identity,
                 data=fingerprint(samples), predictions=fingerprint(all_predictions), max_tokens=settings['max_tokens'])
    out.mkdir(parents=True,exist_ok=True)
    path=out/'judge_config.json'
    if path.exists() and json.loads(path.read_text())!=binding:
        raise ValueError('Judge inputs changed; use a fresh output')
    write_json(path,binding); history=out/'judgments.jsonl';recover_jsonl_tail(history)
    cache={r['cache_key']:r for r in read_jsonl(history) if r['status']=='complete'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoints=settings.get('base_urls', [settings['base_url']])
    endpoint_counter=itertools.count()
    endpoint_lock=threading.Lock()
    def request(cfg, key, prompt, payload):
        body=dict(model=cfg['model'],temperature=0,max_tokens=settings['max_tokens'],
                  chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_object'},
                  messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
        with endpoint_lock:
            endpoint=endpoints[next(endpoint_counter) % len(endpoints)]
        req=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
            headers={'Content-Type':'application/json','Authorization':'Bearer local'},data=json.dumps(body).encode())
        with opener.open(req,timeout=cfg['timeout']) as response:answer=json.load(response)
        choice=answer['choices'][0]
        return choice['message'].get('content') or '',answer.get('usage',{}),choice.get('finish_reason')=='stop'
    results=[];pending=[]
    for row in rows:
        row['cache_key']=fingerprint(dict(binding=binding,row=row))
        if row['cache_key'] in cache:results.append(cache[row['cache_key']])
        else:pending.append(row)
    with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
        futures=[pool.submit(evaluate,row,transport,'local',system,request) for row in pending]
        for future in as_completed(futures):
            row=future.result();append_jsonl(history,row);results.append(row)
    metrics=content_summary(results,negatives,len(samples),sum(r['status']=='complete' for r in all_predictions))
    write_json(out/'judge_metrics.json',metrics)
    return metrics['status']


def group_scores(samples, rows):
    metrics=summarize(samples,rows)
    first_hits=later_hits=first_windows=later_windows=0
    for score in metrics['per_sample']:
        refs=sorted(reference_windows(samples[score['sample_index']]),key=lambda item:(item[1]['window_start'],item[0]))
        matched={m['response_index'] for m in score['matches']}
        first_windows+=1;first_hits+=refs[0][0] in matched
        later_windows+=len(refs)-1;later_hits+=sum(ri in matched for ri,_ in refs[1:])
    metrics['first_later']=dict(first_hits=first_hits,first_windows=first_windows,later_hits=later_hits,later_windows=later_windows)
    minutes=sum(samples[r['sample_index']]['duration_seconds'] for r in rows)/60
    metrics['fp_per_minute']=sum(r['extra_predictions'] for r in metrics['per_sample'])/minutes if minutes else None
    for dimension, key in (('task','task'),('cognitive','cognitive_level'),('length','duration_seconds')):
        def group(row):
            value=samples[row['sample_index']][key]
            if dimension=='length':return '≤60s' if value<=60 else '60–180s' if value<=180 else '180–300s' if value<=300 else '>300s'
            return value
        for name in sorted({group(row) for row in rows}):
            metrics['groups'][dimension+':'+name]=summarize(samples,[r for r in rows if group(r)==name])['groups']['overall']
    return metrics


def report(path):
    config=json.loads(Path(path).read_text());root=Path(config['output']);samples=json.loads(Path(config['data']).read_text())
    models={};judge={}
    for job in config['models']:
        label=job['label'];directory=root/'runs'/label
        if not (directory/'predictions.jsonl').is_file():continue
        rows=predictions(directory);complete=[r for r in rows if r['status']=='complete']
        metrics=group_scores(samples,complete)
        metrics['inference_statuses']=dict(Counter(r['status'] for r in rows))
        metrics['missing_qa']=len(samples)-len(rows)
        scored=metrics['per_sample'];ranked=sorted(scored,key=lambda x:x['extra_predictions'],reverse=True)
        fp=sum(x['extra_predictions'] for x in scored)
        metrics['fp_diagnostics']=dict(top10_share=sum(x['extra_predictions'] for x in ranked[:10])/fp if fp else 0,
                                      worst_qa=ranked[:10])
        write_json(directory/'aoi_report_metrics.json',metrics);models[label]=dict(rows={r['sample_index']:r for r in complete},metrics=metrics)
        jp=root/'judge'/label/'judge_metrics.json'
        if jp.exists():judge[label]=json.loads(jp.read_text())
    common=set.intersection(*(set(m['rows']) for m in models.values())) if models else set()
    excluded=[]
    for i in sorted(common):
        entries=[m['rows'][i] for m in models.values()]
        if len({r.get('system_prompt_sha256') for r in entries})!=1 or max(r['observed_seconds'] for r in entries)-min(r['observed_seconds'] for r in entries)>1e-5:
            excluded.append(i)
    common-=set(excluded)
    paired={label:group_scores(samples,[m['rows'][i] for i in sorted(common)]) for label,m in models.items()}
    lines=['# OmniPro under AOI evaluation','',
        'Timing-only inclusive ±3s one-to-one windows; content scored separately using AOI 0–2 rubric. These are not original paper content-gated F1 scores.',
        f"Source QA: {len(samples)}. Paired valid QA across available models: {len(common)}. Prompt/duration mismatches: {len(excluded)}.",'',
        '| Model | Complete / error / missing QA | Recall | Precision | F1 | TP content mean /2 | All-window mean /2 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'N/A' if v is None else f'{v:.3f}'
    for label,m in models.items():
        g=paired[label]['groups'].get('overall',{});j=judge.get(label,{}).get('groups',{}).get('overall',{})
        status=m['metrics']['inference_statuses'];num=lambda k:fmt(100*g[k]) if g.get(k) is not None else 'N/A'
        lines.append(f"| {label} | {status.get('complete',0)} / {sum(v for k,v in status.items() if k!='complete')} / {m['metrics']['missing_qa']} | {num('window_hit_rate')} | {num('timing_precision')} | {num('timing_f1')} | {fmt(j.get('response_mean_matched'))} | {fmt(j.get('response_mean_all_windows'))} |")
    lines+=['','Content aggregates above use each model’s own completed matches; denominators differ. Consult judge_metrics.json for scored, failed and matched window counts. Timing comparisons use the same valid QA intersection.','']
    for prefix in ('task:','cognitive:','length:','mode:'):
        lines += ['## '+prefix[:-1],'','| Group | '+' | '.join(models)+' |','|---|'+'---:|'*len(models)]
        for group in sorted({k for m in paired.values() for k in m['groups'] if k.startswith(prefix)}):
            cells=[]
            for label in models:
                g=paired[label]['groups'].get(group,{})
                cells.append(f"{100*g['timing_f1']:.2f} (n={g['samples']})" if g else 'N/A')
            lines.append('| '+group+' | '+' | '.join(cells)+' |')
        lines.append('')
    manifest_path=Path(config['data']).parent/'manifest.json'
    if manifest_path.exists():
        audit=json.loads(manifest_path.read_text())
        lines += ['## Input integrity','',
            f"Source references: {audit['references']}. Triggers beyond actual video end: {len(audit['anomalies'])}. Unreachable ±3s windows retained: {sum(r['window_unreachable'] for r in audit['anomalies'])}.",
            'Source time strings differing from numeric timestamps: '+str(len(audit.get('timestamp_annotations',[])))+'. See inputs/manifest.json.', '']
    atomic_write_text(root/'comparison.md','\n'.join(lines)+'\n')
    write_json(root/'comparison.json',dict(paired_qa=len(common),excluded_mismatches=excluded,
        models={k:v['metrics'] for k,v in models.items()},paired=paired,judge=judge))
