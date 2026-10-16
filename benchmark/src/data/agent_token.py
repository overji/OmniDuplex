"""Versioned Agent separator protocol and lossless native token-stream parsing."""
import hashlib
import json
from pathlib import Path

TOKEN = '<|agent|>'
PROTOCOL = 'agent_token_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokenizer_identity(tokenizer, *, exclude_agent=False):
    vocab = tokenizer.get_vocab()
    if exclude_agent:
        vocab = {k: v for k, v in vocab.items() if k != TOKEN}
    return digest(vocab)


def base_identity(path):
    path = Path(path)
    files = ['config.json', 'model.safetensors.index.json']
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in files if (path / name).is_file()}


def adapter_identity(path):
    path = Path(path)
    required = ['agent_protocol.json', 'agent_rows.safetensors', 'adapter_config.json',
                'adapter_model.safetensors', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']
    required = sorted(set(required) | {str(f.relative_to(path)) for f in (path / 'tokenizer').rglob('*') if f.is_file()})
    result = {}
    for name in required:
        file = path / name
        if not file.is_file():
            raise ValueError(f'Incomplete agent-token adapter: {name}')
        h = hashlib.sha256()
        with file.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                h.update(block)
        result[name] = h.hexdigest()
    meta = json.loads((path / 'agent_protocol.json').read_text())
    if meta.get('protocol') != PROTOCOL or meta.get('token') != TOKEN:
        raise ValueError('Unsupported adapter protocol')
    return result


def full_model_identity(path):
    """Export identity plus config/index/tokenizer checks, not a weight-byte hash."""
    from transformers import AutoTokenizer
    path = Path(path)
    meta = json.loads((path / 'full_training_model.json').read_text())
    if not (path / 'config.json').is_file():
        raise ValueError('Incomplete full-model export: config.json')
    index = path / 'model.safetensors.index.json'
    shards = (set(json.loads(index.read_text())['weight_map'].values()) if index.is_file()
              else {'model.safetensors'})
    if not all((path / shard).is_file() for shard in shards):
        raise ValueError('Incomplete full-model export: missing weight shard')
    for name, expected in meta.get('runtime_files', {}).items():
        file = path / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Full-model runtime code/processor mismatch: ' + name)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if (meta.get('protocol') != PROTOCOL or meta['files'] != base_identity(path)
            or tokenizer_identity(tokenizer) != meta['tokenizer_sha256']
            or tokenizer.encode(TOKEN, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(TOKEN)]):
        raise ValueError('Full-model export/tokenizer identity mismatch')
    return meta


class AgentStream:
    """A turn survives media chunks; only a complete, valid turn exposes a call."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.agent_id = tokenizer.convert_tokens_to_ids(TOKEN)
        if tokenizer.encode(TOKEN, add_special_tokens=False) != [self.agent_id] or TOKEN not in tokenizer.get_vocab():
            raise ValueError('Missing atomic Agent token')
        self.controls = {tokenizer.convert_tokens_to_ids(t) for t in (
            '<|speak|>', '<|listen|>', '<|turn_eos|>', '<|chunk_eos|>', '</unit>')}
        self.reset()

    def reset(self):
        self.raw_ids, self.response_ids, self.call_ids = [], [], []
        self.onset = self.call_onset = None
        self.separators = 0
        self.last_time = None

    def feed(self, ids, clock, *, ended=False, interrupted=False):
        for token in ids:
            self.raw_ids.append(int(token))
            if token == self.agent_id:
                self.separators += 1
                if self.call_onset is None:
                    self.call_onset = clock
                if self.onset is None:
                    self.onset = clock
            elif token not in self.controls and token not in self.tokenizer.all_special_ids:
                if self.onset is None:
                    self.onset = clock
                (self.call_ids if self.separators else self.response_ids).append(int(token))
                self.last_time = clock
        if not (ended or interrupted):
            return None
        return self.finish(complete=ended and not interrupted, clock=clock)

    def finish(self, *, complete=False, clock=None):
        if self.onset is None:
            self.reset()
            return None
        response = self.tokenizer.decode(self.response_ids, skip_special_tokens=False)
        call_text = self.tokenizer.decode(self.call_ids, skip_special_tokens=False)
        calls, status = [], 'absent'
        if self.separators:
            status = 'truncated' if not complete else 'malformed'
            if complete and self.separators == 1:
                try:
                    obj = json.loads(call_text)
                    if (not isinstance(obj, dict) or obj.get('name') != 'agent_call'
                            or not isinstance(obj.get('arguments'), dict)):
                        raise ValueError('Invalid agent_call envelope')
                    calls = [{'type': 'function', 'function': obj}]
                    status = 'structured'
                except (ValueError, TypeError):
                    pass
            if self.separators > 1:
                status = 'duplicate_separator'
        event = {'onset_seconds': self.onset, 'text': response + (TOKEN + call_text if self.separators else ''),
            'response_text': response, 'tool_calls': calls, 'agent_parse_status': status,
            'agent_call_text': call_text, 'agent_attempted': bool(self.separators),
            'agent_call_onset_seconds': self.call_onset,
            'agent_call_complete_seconds': (clock if clock is not None else self.last_time) if calls else None,
            'raw_generated_ids': self.raw_ids[:], 'response_empty': not response.strip()}
        self.reset()
        return event
