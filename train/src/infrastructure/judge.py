"""Rank-zero, reference-only semantic judging. No model is loaded in training ranks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

from benchmark.data.files import write_json
from benchmark.evaluation.judge import prepare_rows, validate_answer


VERSION = 'grpo-reference-judge-silence-v1'


class JudgeFailure(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def identity(settings):
    """Bind caches/resume to local model metadata, shard stats, prompt and decoding."""
    root = Path(settings['model_path'])
    paths = [root / name for name in ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json')]
    if not all(p.is_file() for p in paths):
        raise ValueError('Judge model metadata is incomplete')
    return {'version': VERSION, 'model': settings['model'],
            'metadata': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'shards': [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(root.glob('*.safetensors'))],
            'prompt': hashlib.sha256(Path(settings['system_prompt']).read_bytes()).hexdigest(),
            'max_tokens': settings['max_tokens'], 'temperature': 0, 'enable_thinking': False}


class Judge:
    def __init__(self, settings):
        self.settings = settings
        self.identity = identity(settings)
        self.system = Path(settings['system_prompt']).read_text()
        self.root = Path(settings['cache_dir']) / fingerprint(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.key_locks = {}
        self.stats = dict(requests=0, score_requests=0, cache_hits=0, failures=0, retries=0)
        # Explicitly bypass environment proxies for this localhost-only transport.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path, body=None):
        key = os.environ.get(self.settings.get('api_key_env', ''), 'local')
        request = urllib.request.Request(self.settings['base_url'].rstrip('/') + path,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            data=None if body is None else json.dumps(body).encode())
        with self.opener.open(request, timeout=self.settings['timeout']) as response:
            return json.load(response)

    def ready(self):
        models = self._request('/models')
        if self.settings['model'] not in {r['id'] for r in models['data']}:
            raise JudgeFailure('Configured judge model is not served')

    def snapshot(self):
        with self.lock:
            return dict(self.stats)

    def _increment(self, key):
        with self.lock:
            self.stats[key] += 1

    def _score(self, row, cancelled=None):
        payload = row['payload']
        if row['event_index'] is None or (not payload['candidate_response'].strip() and not payload['judge_call']):
            return dict(status='complete', response_score=0, call_score=0 if row['call_applicable'] else None)
        self._increment('score_requests')
        key = fingerprint({'identity': self.identity, 'payload': payload})
        with self.lock:
            lock = self.key_locks.setdefault(key, threading.Lock())
        with lock:
            if cancelled is not None and cancelled.is_set():
                raise JudgeFailure('Candidate group was cancelled')
            path = self.root / (key + '.json')
            if path.exists():
                cached = json.loads(path.read_text())
                if cached['identity'] != self.identity or cached['payload'] != payload:
                    raise JudgeFailure('Judge cache identity changed')
                answer = validate_answer(json.dumps(cached['answer']), payload['judge_call'])
                self._increment('cache_hits')
                return self._finish(row, answer)
            attempts = []
            for attempt in range(self.settings['retries'] + 1):
                if cancelled is not None and cancelled.is_set():
                    raise JudgeFailure('Candidate group was cancelled')
                if attempt:
                    self._increment('retries')
                    time.sleep(min(2 ** (attempt - 1), 8))
                self._increment('requests')
                audit = {'attempt': attempt + 1}
                try:
                    body = self._request('/chat/completions', {
                        'model': self.settings['model'], 'temperature': 0,
                        'max_tokens': self.settings['max_tokens'],
                        'chat_template_kwargs': {'enable_thinking': False},
                        'response_format': {'type': 'json_object'},
                        'messages': [{'role': 'system', 'content': self.system},
                                     {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]})
                    choice = body['choices'][0]
                    text = choice['message'].get('content') or ''
                    audit.update(raw_response=text, usage=body.get('usage'), finish_reason=choice['finish_reason'])
                    if choice['finish_reason'] != 'stop':
                        raise ValueError('Judge response was truncated')
                    answer = validate_answer(text, payload['judge_call'])
                    attempts.append(audit)
                    write_json(path, {'identity': self.identity, 'payload': payload, 'answer': answer,
                                      'attempts': attempts})
                    return self._finish(row, answer)
                except Exception as exc:
                    # Never persist provider exception text, which can contain credentials.
                    audit['error_type'] = type(exc).__name__
                    attempts.append(audit)
            self._increment('failures')
            write_json(self.root / (key + '.failed.json'), {'identity': self.identity, 'payload': payload,
                                                         'attempts': attempts})
            raise JudgeFailure('Judge retries exhausted; discard the entire candidate group')

    @staticmethod
    def _finish(row, answer):
        answer = dict(answer)
        if not row['payload']['candidate_response'].strip():
            answer['response_score'] = 0
        if row['call_applicable'] and not row['payload']['judge_call']:
            answer['call_score'] = 0
        return dict(answer, status='complete')

    def score_group(self, sample, trajectories):
        from train.src.domain.agent import training_sample
        from benchmark.data.prompts import TOOLS, describe_tool
        sample = training_sample(sample)
        work = []
        for candidate, trajectory in enumerate(trajectories):
            prediction = dict(sample_index=0, status='complete', events=trajectory['events'],
                              **{k: sample.get(k) for k in ('task', 'video', 'question')})
            rows, _ = prepare_rows([sample], [prediction])
            for row in rows:
                available = sample.get('available_tools', [])
                row['payload']['available_tool_definitions'] = {name: describe_tool(name) for name in available if name in TOOLS}
                work.append((candidate, row))
        # Preserve stable ordering, but cancel queued work on the first failure.
        # Otherwise a dead server could keep every training rank waiting through
        # many waves of retries, exceeding distributed collective timeouts.
        cancelled = threading.Event()
        answers = [None] * len(work)
        with ThreadPoolExecutor(max_workers=self.settings['concurrency']) as pool:
            futures = {pool.submit(self._score, row, cancelled): index for index, (_, row) in enumerate(work)}
            try:
                for future in as_completed(futures):
                    answers[futures[future]] = future.result()
            except Exception:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        results = [{} for _ in trajectories]
        for (candidate, row), answer in zip(work, answers):
            results[candidate][row['response_index']] = answer
        return results
