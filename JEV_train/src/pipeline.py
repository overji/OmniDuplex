"""One consumer owns both caches. The producer only queues immutable media units."""
from queue import Queue
import json
import threading
import time

from .config import write


def model_view(sample):
    return {key:sample[key] for key in ('id','question','question_time','mode','task','ability',
        'available_tools','frontend_context','_agent_call_schema') if key in sample}


def run_session(row, source, gate, backend, realtime=False,on_progress=None):
    queue=Queue();stop=threading.Event()
    def produce():
        origin=time.monotonic()
        try:
            for unit in source:
                if stop.is_set():break
                if realtime and stop.wait(max(0,origin+unit.second-time.monotonic())):break
                queue.put(unit)
        except Exception as exc:queue.put(exc)
        finally:queue.put(None)
    producer=threading.Thread(target=produce,daemon=True)
    events=[];probabilities=[];trace=[]
    try:
        sample=model_view(row['sample']);gate.start(sample);backend.start_session(sample);producer.start()
        while True:
            unit=queue.get()
            if unit is None:break
            if isinstance(unit,Exception):raise unit
            before=time.monotonic();decision,p=gate.decide(unit);gate_seconds=time.monotonic()-before
            before=time.monotonic();backend.append_media(unit);prefill_seconds=time.monotonic()-before
            probabilities.append(p)
            item=dict(second=unit.second,response=decision,p_response=p,gate_seconds=gate_seconds,
                      backend_prefill_seconds=prefill_seconds,queue_depth=queue.qsize())
            if decision:
                before=time.monotonic();result=backend.generate_response();item['generation_seconds']=time.monotonic()-before
                events.append(dict(onset_seconds=float(unit.second),**result))
                gate.feedback(result['text'])
            trace.append(item)
            if on_progress:on_progress(dict(sample_index=row['sample_index'],events=list(events),
                                            probabilities=list(probabilities),trace=list(trace)))
        if len(probabilities)!=row['horizon']:raise RuntimeError('Media ended before expected horizon')
        return dict(sample_index=row['sample_index'],status='complete',events=events,probabilities=probabilities,trace=trace,
                    cache_verified=getattr(backend,'checked',None))
    finally:
        stop.set()
        if producer.is_alive():producer.join(timeout=2)
        backend.close_session()


def run(rows,config,output,realtime=False):
    import torch
    from .gate import load,GateSession
    from .backend import BACKENDS
    from .media import units
    from .metrics import evaluate
    if output.exists():raise ValueError('Use a fresh evaluation directory')
    if not config.get('classifier_adapter'):raise ValueError('A trained classifier_adapter is required')
    state=load(config);gate=GateSession(state,config);backend=BACKENDS[config['backend']](config)
    predictions=[];output.mkdir(parents=True)
    write(output/'run_config.json',{**config,'agent_call_schema':'task-start'})
    for row in rows:
        partial={}
        try:
            with torch.inference_mode():result=run_session(row,units(row,config),gate,backend,realtime,partial.update)
        except Exception as exc:
            result=dict(**partial,status='failed',error=f'{type(exc).__name__}: {exc}')
            result['sample_index']=row['sample_index']
        result.update(task=row['sample']['task'],video=row['sample']['video'],question=row['sample']['question'])
        predictions.append(result)
        write(output/'predictions.json',predictions)
        from benchmark.data.files import atomic_write_text
        atomic_write_text(output/'predictions.jsonl',''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions))
        write(output/'metrics.json',evaluate(rows,predictions,config['threshold']))
    metrics=evaluate(rows,predictions,config['threshold'])
    lines=['# JEV evaluation','',f"Completed: {metrics['completed_qa']}/{metrics['expected_qa']}",'',
           'Logical trigger timing uses the current media end timestamp; generation wall time is separate.','',
           '```json',__import__('json').dumps({k:v for k,v in metrics.items() if k!='timing'},indent=2),'```','',
           'Content correctness is not automatically judged.']
    lines+=['','| Group | QA | Windows | Triggers | Hits | Recall | Precision | F1 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,score in metrics['timing']['groups'].items():
        rate=score['window_hit_rate']
        formatted=f'{rate:.2%}' if rate is not None else 'N/A'
        lines.append(f"| {name} | {score['samples']} | {score['windows']} | {score['predictions']} | {score['hits']} | "
            f"{formatted} | {score['timing_precision']:.2%} | {score['timing_f1']:.2%} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
