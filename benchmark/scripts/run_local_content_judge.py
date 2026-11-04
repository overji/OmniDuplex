#!/usr/bin/env python3
"""Judge existing predictions with the same local replicas used for OmniPro."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.data.files import read_jsonl,write_json,atomic_write_text
from benchmark.runtime.omnipro_queue import Queue


def normalized_report(config):
    """Normalize individual 0..2 scores before averaging; failures stay missing."""
    root=Path(config['output']);summary={};matched={}
    lines=['# AOI-Bench local content Judge','',
           'Scores are individually divided by 2 before averaging. Unmatched windows score zero; failed Judge requests remain missing.',
           'TP content means are conditional on each model\'s own timing matches.','',
           '| Model | Group | Windows | Matched | Failed | TP response | All response | All task |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def mean(values):return sum(values)/len(values) if values else None
    def fmt(value):return 'N/A' if value is None else f'{100*value:.2f}%'
    for job in config['models']:
        label=job['label'];path=root/'judge'/label/'judgments.jsonl'
        if not (root/'judge'/label/'judge_metrics.json').exists():continue
        # Last attempt owns the terminal result when resuming a failed request.
        rows=list({(r['sample_index'],r['response_index']):r for r in read_jsonl(path,complete_only=True)}.values())
        groups=defaultdict(list)
        matched[label]={(r['sample_index'],r['response_index']):r for r in rows if r['status']=='complete' and r['event_index'] is not None}
        for row in rows:
            groups['overall'].append(row);groups['mode:'+row['mode']].append(row)
        summary[label]={}
        for name,items in groups.items():
            ok=[r for r in items if r['status']=='complete'];tp=[r for r in items if r['event_index'] is not None]
            tp_ok=[r for r in tp if r['status']=='complete']
            response=[r['response_score']/2 for r in ok]
            task=[(r['response_score']+r['call_score'])/4 if r['call_applicable'] else r['response_score']/2 for r in ok]
            value=dict(windows=len(items),matched=len(tp),failed=len(items)-len(ok),
                response_mean_matched=mean([r['response_score']/2 for r in tp_ok]) if len(tp_ok)==len(tp) else None,
                response_mean_all=mean(response) if len(ok)==len(items) else None,
                task_mean_all=mean(task) if len(ok)==len(items) else None)
            summary[label][name]=value
            lines.append(f"| {label} | {name} | {len(items)} | {len(tp)} | {value['failed']} | {fmt(value['response_mean_matched'])} | {fmt(value['response_mean_all'])} | {fmt(value['task_mean_all'])} |")
    common=set.intersection(*(set(r) for r in matched.values())) if len(matched)>1 else set()
    paired={label:mean([rows[k]['response_score']/2 for k in common]) for label,rows in matched.items()} if common else {}
    if common:
        lines+=['',f'Both/all models matched the same {len(common)} windows: '+', '.join(f'{label}: {fmt(value)}' for label,value in paired.items())+'.']
    write_json(root/'normalized_summary.json',dict(scale=[0,1],models=summary,shared_matched_windows=len(common),shared_matched_response_means=paired))
    atomic_write_text(root/'comparison.md','\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();queue=Queue(args.config);config=queue.config
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (queue.root/'.local_judge.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            queue.judge_models(config['models'],lambda:normalized_report(config))
            queue.state('complete')
            # Resume the saved audit only after Judge services have released GPUs.
            if config.get('continuation'):
                with (queue.root/'continuation.log').open('ab') as log:
                    child=subprocess.Popen(config['continuation'],cwd=config['continuation_cwd'],
                        env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(queue.root/'continuation.json',dict(pid=child.pid,argv=config['continuation']))
        except KeyboardInterrupt:queue.state('stopped')
        except BaseException as exc:queue.state('failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:queue.stop_children()


if __name__=='__main__':main()
