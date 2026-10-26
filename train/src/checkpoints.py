"""Plain-text Agent LoRA/full-export identity; tokenizer and vocabulary stay unchanged."""
import hashlib
import json
from pathlib import Path

from benchmark.data.agent_token import base_identity, tokenizer_identity
from train.src.agent import PROTOCOL, SEPARATOR


def tokenizer_metadata(tokenizer):
    ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    if not ids or getattr(tokenizer, 'unk_token_id', None) in ids or '<|agent|>' in tokenizer.get_vocab():
        raise ValueError('Use the original tokenizer with an encodable task_start literal, not old Agent-token weights')
    return {'protocol': PROTOCOL, 'agent_call_schema': 'task-start', 'separator': SEPARATOR,
            'separator_token_ids': ids, 'vocab_size': len(tokenizer),
            'tokenizer_sha256': tokenizer_identity(tokenizer)}


def adapter_identity(path):
    path = Path(path)
    meta = json.loads((path / 'agent_protocol.json').read_text())
    if meta.get('protocol') != PROTOCOL or meta.get('separator') != SEPARATOR or meta.get('agent_call_schema') != 'task-start':
        raise ValueError('Unsupported adapter protocol; retrain using plain-text instruction targets')
    if (path / 'agent_rows.safetensors').exists():
        raise ValueError('Legacy Agent row weights are incompatible')
    files = ['agent_protocol.json', 'adapter_config.json', 'adapter_model.safetensors',
             'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']
    files = sorted(set(files) | {str(p.relative_to(path)) for p in (path / 'tokenizer').rglob('*') if p.is_file()})
    result = {}
    for name in files:
        p = path / name
        if not p.is_file():
            raise ValueError(f'Incomplete adapter: {name}')
        result[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def save_adapter(llm, tokenizer, base_path, path):
    path = Path(path)
    meta = {**tokenizer_metadata(tokenizer), 'base_identity': base_identity(base_path)}
    llm.save_pretrained(path, save_embedding_layers=False)
    tokenizer.save_pretrained(path / 'tokenizer')
    (path / 'agent_protocol.json').write_text(json.dumps(meta, indent=2) + '\n')
    adapter_identity(path)


def load_adapter(model, base_path, path):
    from transformers import AutoTokenizer
    from peft import PeftModel
    path = Path(path)
    adapter_identity(path)
    meta = json.loads((path / 'agent_protocol.json').read_text())
    original = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path / 'tokenizer'), trust_remote_code=True, local_files_only=True)
    expected = {**tokenizer_metadata(tokenizer), 'base_identity': base_identity(base_path)}
    if meta != expected or tokenizer_identity(original) != tokenizer_identity(tokenizer):
        raise ValueError('Adapter base/tokenizer identity mismatch')
    if model.llm.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError('Model vocabulary mismatch')
    model.llm = PeftModel.from_pretrained(model.llm, str(path)).merge_and_unload()
    model.prepare_processor(tokenizer=tokenizer)
    return meta


def full_model_identity(path):
    from transformers import AutoTokenizer
    path = Path(path)
    meta = json.loads((path / 'full_training_model.json').read_text())
    if meta.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported full-model protocol; retrain with plain-text instruction targets')
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    expected = tokenizer_metadata(tokenizer)
    if any(meta.get(k) != v for k, v in expected.items()) or meta.get('files') != base_identity(path):
        raise ValueError('Full-model tokenizer/config identity mismatch')
    index = path / 'model.safetensors.index.json'
    shards = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    if not all((path / name).is_file() for name in shards):
        raise ValueError('Full-model weights incomplete')
    if any(not (path / name).is_file() or hashlib.sha256((path / name).read_bytes()).hexdigest() != h
           for name, h in meta.get('runtime_files', {}).items()):
        raise ValueError('Full-model runtime files changed')
    return meta
