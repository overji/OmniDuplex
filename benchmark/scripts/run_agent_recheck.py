#!/usr/bin/env python3
"""Supervise Agent-only reevaluation and report actual, schema-checked calls."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmark.data.files import read_jsonl, write_json, atomic_write_text
from benchmark.data.prompts import available_function_schemas
from benchmark.data.agent_schema import adapt_sample, validate_task_start
from benchmark.evaluation.agent_calls import parse_event
from benchmark.runtime.parallel import terminate_owned


def option(command, flag, value):
    command = list(command)
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        command += [flag, str(value)]
    return command


def call_metrics(samples, directory, expected, validate):
    counts = Counter()
    for row in read_jsonl(directory/'predictions.jsonl', complete_only=True):
        if row['status'] != 'complete':
            continue
        sample = samples[row['sample_index']]
        counts['completed_qa'] += 1
        parsed_calls, schema_ok = False, False
        for event in row['events']:
            parsed = parse_event(event)
            counts['output_events'] += 1
            counts['call_attempt_events'] += parsed['attempted']
            if not parsed['calls']:
                continue
            parsed_calls = True
            counts['parseable_calls'] += len(parsed['calls'])
            validation = validate(parsed['calls'], available_function_schemas(sample))
            simple = sample.get('_agent_call_schema') == 'task-start'
            valid = validation.valid and (not simple or all(validate_task_start(c) for c in parsed['calls']))
            if parsed['status'] != 'malformed' and valid:
                schema_ok = True
                counts['schema_valid_call_events'] += 1
                if simple:
                    continue
                for call in parsed['calls']:
                    args = call['function']['arguments']
                    suggested = set(args.get('suggested_capabilities', []))
                    available = set(sample.get('available_tools', []))
                    counts['calls_using_only_available_capabilities'] += bool(suggested) and suggested <= available
                    counts['calls_covering_reference_required_tools'] += bool(suggested) and suggested <= available and set(sample.get('required_tools', [])) <= suggested
        counts['qa_with_parseable_calls'] += parsed_calls
        counts['qa_with_schema_valid_calls'] += schema_ok
    return {'expected_qa': expected, **dict(counts),
            'qa_call_rate': counts['qa_with_parseable_calls']/counts['completed_qa'] if counts['completed_qa'] else None,
            'note': 'Syntax/schema and capability checks only; no execution or semantic LLM judging.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commands', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', required=True)
    p.add_argument('--max-tool-tokens', type=int, default=1024)
    p.add_argument('--agent-call-schema', choices=('task-start', 'legacy'), default='task-start')
    p.add_argument('--dry-run', action='store_true', help='Write the immutable plan without starting inference')
    a = p.parse_args()
    if a.max_tool_tokens < 1 or len(set(a.gpus)) != len(a.gpus):
        p.error('Positive tool budget and distinct GPU indices required')
    out = a.output_dir.resolve();out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.agent_recheck.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries = json.loads(a.commands.read_text())
    prepared = {n:next(e['argv'] for e in entries if e['mode']=='realtime' and e['adapter']==n) for n in ('minicpmo','gander')}
    base=prepared['minicpmo'];data=Path(base[base.index('--data')+1]);videos=Path(base[base.index('--videos-dir')+1]);samples=[adapt_sample(s,a.agent_call_schema) for s in json.loads(data.read_text())]
    indices=[i for i,s in enumerate(samples) if s.get('mode')=='agent' and s.get('sample_type')=='positive' and (videos/s['video']).is_file()]
    if not indices:raise ValueError('No available positive Agent QA')
    index_path=out/'agent_indices.json';write_json(index_path,indices)
    source=Path(prepared['gander'][prepared['gander'].index('--official-code')+1])/'minicpm_ft'
    sys.path.insert(0,str(source))
    from mcpmft.tool_protocol import validate_tool_calls
    commands=[]
    for name,argv in prepared.items():
        cmd=option(argv,'--output-dir',out/name);cmd[cmd.index('run')]='parallel'
        cmd=option(cmd,'--agent-call-schema',a.agent_call_schema)
        cmd+=['--gpus',*a.gpus,'--sample-indices',str(index_path)]
        if name=='minicpmo':cmd+=['--agent-output-protocol','structured']
        else:cmd=option(cmd,'--max-tool-tokens',a.max_tool_tokens)
        commands.append({'model':name,'argv':cmd})
    hashes={str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        REPO/'benchmark/configs/prompts/agent_structured.txt',REPO/'benchmark/configs/prompts/gander_agent.txt',
        REPO/'benchmark/src/evaluation/agent_calls.py',REPO/'benchmark/src/adapters/minicpmo.py',REPO/'benchmark/src/adapters/gander.py',
        REPO/'benchmark/src/data/agent_schema.py',REPO/'benchmark/configs/prompts/agent_task_start_structured.txt',
        REPO/'benchmark/configs/prompts/agent_task_start_gander.txt', REPO/'benchmark/src/data/agent_inputs.py',
        REPO/'benchmark/src/data/prompts.py', REPO/'benchmark/src/runtime/runner.py')}
    plan={'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'sample_indices':indices,'commands':commands,'protocol_hashes':hashes,'gpus':a.gpus,'agent_call_schema':a.agent_call_schema}
    path=out/'recheck_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Recheck protocol changed; use a new output directory')
    write_json(path,plan)
    if a.dry_run:
        print(f'Prepared {len(indices)} Agent QA per model, schema={a.agent_call_schema}; no inference started')
        return
    state={'pid':os.getpid(),'status':'running','agent_qa':len(indices),'jobs':[]};processes=[]
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def save():write_json(out/'recheck_status.json',state)
    try:
        for item in commands:
            name=item['model'];directory=out/name;directory.mkdir(exist_ok=True)
            job={'model':name,'status':'running','output':str(directory)};state['jobs'].append(job)
            env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1'}
            with (directory/'supervisor.log').open('ab') as log:
                child=subprocess.Popen(item['argv'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(child);job['pid']=child.pid;save();print('START',name,child.pid,flush=True)
                while child.poll() is None:
                    job['calls']=call_metrics(samples,directory,len(indices),validate_tool_calls);save();time.sleep(15)
                job.update(status='complete' if child.returncode==0 else 'incomplete',exit_code=child.returncode,
                           calls=call_metrics(samples,directory,len(indices),validate_tool_calls))
                write_json(directory/'agent_call_metrics.json',job['calls']);processes.remove(child);save()
                print('END',name,job['status'],flush=True)
        state['status']='complete' if all(j['status']=='complete' for j in state['jobs']) else 'incomplete'
    except KeyboardInterrupt:
        state['status']='interrupted'
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        terminate_owned(processes);save()


if __name__=='__main__':main()
