"""Text-only AOI Judge and deterministic T1 tool-name scoring."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import itertools
from pathlib import Path
import urllib.request

from benchmark.data.files import append_jsonl, read_jsonl, recover_jsonl_tail, write_json
from benchmark.data.interactscope import tool_names
from benchmark.evaluation.judge import prepare_rows, evaluate, fingerprint


def predicted_tools(event):
    """Keep unknown explicit names so extra/wrong tools cannot disappear."""
    text = event.get('response_text', event.get('text', ''))
    names = tool_names(text)
    explicit = list(event.get('tool_calls') or [])
    # These are explicit call/name containers, not mentions extracted from prose.
    # No fuzzy name correction and no inference from task_start instructions.
    import re
    payloads = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    payloads.append(text.strip())
    for payload in payloads:
        try:
            value = json.loads(payload)
        except (ValueError,TypeError):
            continue
        if isinstance(value,dict):
            value = value.get('tool_calls',[value])
        if isinstance(value,list):
            for call in value:
                if isinstance(call,str):
                    names.append(call)
                elif isinstance(call,dict):
                    explicit.append(call)
    for call in explicit:
        if not isinstance(call,dict):
            names.append('<invalid-call>')
            continue
        function = call.get('function', call)
        if isinstance(function, dict) and isinstance(function.get('name'), str):
            names.append(function['name'])
        else:
            names.append('<invalid-call>')
    return sorted(set(names))


def tool_judgment(row, sample, prediction):
    index = row['event_index']
    names = predicted_tools(prediction['events'][index]) if index is not None else []
    gold = sample['responses'][row['response_index']]['reference_tool_names']
    correct = index is not None and bool(gold) and names == sorted(set(gold))
    return {k:v for k,v in row.items() if k!='payload'} | dict(status='complete', content_kind='tool_name_set',
        response_score=2 if correct else 0, call_score=None, call_applicable=False,
        response_reason='Exact tool-name set match' if correct else 'Missing, extra, wrong or unmatched tool name set',
        call_reason='Not applicable; source has no instruction annotation', error_types=[] if correct else ['tool_names'],
        predicted_tool_names=names, reference_tool_names=gold, usage_attempts=[])


def judge(config, label, base_url=None):
    root = Path(config['output'])
    samples = json.loads(Path(config['data']).read_text())
    predictions = read_jsonl(root/'runs'/label/'predictions.jsonl', complete_only=True)
    rows, _ = prepare_rows(samples, predictions)
    by_index = {p['sample_index']:p for p in predictions}
    settings = config['judge']
    endpoints = base_url or settings.get('base_urls') or settings.get('base_url') or []
    if isinstance(endpoints,str):
        endpoints = [endpoints]
    if not endpoints and any(samples[r['sample_index']]['mode']!='tool' and r['event_index'] is not None
                             and r['payload']['candidate_response'].strip() for r in rows):
        raise ValueError('Explicit local Qwen27B endpoint required for Judge')
    system = Path(settings['system_prompt']).read_text()
    binding = dict(protocol=config['protocol'], data=config['data_sha256'], predictions=fingerprint(predictions),
                   judge=dict(model=settings['model'], base_urls=endpoints, max_tokens=settings['max_tokens'],
                              prompt_sha256=fingerprint(system)))
    directory = root/'judge'/label
    directory.mkdir(parents=True, exist_ok=True)
    meta = directory/'judge_config.json'
    if meta.exists() and json.loads(meta.read_text()) != binding:
        # Incremental inference may append predictions. Resume cache keys bind the
        # actual per-window payload; model/data/prompt changes remain forbidden.
        previous = json.loads(meta.read_text())
        if {k:v for k,v in previous.items() if k!='predictions'} != {k:v for k,v in binding.items() if k!='predictions'}:
            raise ValueError('Judge inputs changed; use a new output directory')
    write_json(meta, binding)
    path = directory/'judgments.jsonl'
    recover_jsonl_tail(path)
    cache = {r['cache_key']:r for r in read_jsonl(path) if r['status']=='complete'}
    pending = []
    for row in rows:
        sample = samples[row['sample_index']]
        row['content_kind'] = 'tool_name_set' if sample['mode']=='tool' else 'text'
        row['payload']['user_context'] = sample.get('frontend_context', {}).get('system_prompt', '')
        row['cache_key'] = fingerprint(dict(binding={k:v for k,v in binding.items() if k!='predictions'}, row=row))
        if row['cache_key'] not in cache:
            pending.append(row)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint_counter = itertools.count()
    transport = dict(model=settings['model'], timeout=settings['timeout'], retries=settings['retries'])
    def request(cfg, key, prompt, payload):
        body = dict(model=cfg['model'], temperature=0, max_tokens=settings['max_tokens'],
                    chat_template_kwargs={'enable_thinking':False}, response_format={'type':'json_object'},
                    messages=[dict(role='system', content=prompt), dict(role='user', content=json.dumps(payload,ensure_ascii=False))])
        endpoint = endpoints[next(endpoint_counter) % len(endpoints)]
        req = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
            headers={'Content-Type':'application/json','Authorization':'Bearer local'}, data=json.dumps(body).encode())
        with opener.open(req, timeout=cfg['timeout']) as response:
            answer = json.load(response)
        choice = answer['choices'][0]
        return choice['message'].get('content') or '', answer.get('usage',{}), choice.get('finish_reason')=='stop'
    def score(row):
        sample = samples[row['sample_index']]
        if sample['mode']=='tool':
            return tool_judgment(row, sample, by_index[row['sample_index']])
        return evaluate(row, transport, 'local', system, request)
    with ThreadPoolExecutor(max_workers=settings['concurrency']) as pool:
        for future in as_completed([pool.submit(score, row) for row in pending]):
            record = future.result()
            append_jsonl(path, record)
            cache[record['cache_key']] = record
    # Write only current cache keys; failed attempts remain in immutable JSONL history.
    current = [cache.get(row['cache_key'], dict(row, status='failed')) for row in rows]
    write_json(directory/'current_judgments.json', current)
    return dict(windows=len(current), failed=sum(r['status']!='complete' for r in current))
