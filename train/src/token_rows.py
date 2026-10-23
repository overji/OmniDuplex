"""Train only the new vocabulary row, alongside PEFT; old vocabulary stays frozen."""
import json
from pathlib import Path

import torch
from torch import nn
from benchmark.data.agent_token import TOKEN, PROTOCOL, base_identity, tokenizer_identity, adapter_identity


def mean_row(weight):
    total = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
    for start in range(0, len(weight), 2048):
        total += weight[start:start + 2048].float().sum(0)
    return total / len(weight)


def register_token(model, tokenizer, base_path):
    if TOKEN in tokenizer.get_vocab():
        raise ValueError('Training base already contains Agent token; use the original base')
    old_vocab = len(tokenizer)
    if model.llm.get_input_embeddings().weight.shape[0] != old_vocab:
        raise ValueError('Base tokenizer and embedding vocabulary differ')
    before = tokenizer_identity(tokenizer)
    input_mean = mean_row(model.llm.get_input_embeddings().weight).detach()
    output_mean = mean_row(model.llm.get_output_embeddings().weight).detach()
    tokenizer.add_special_tokens({'additional_special_tokens': [TOKEN]}, replace_additional_special_tokens=False)
    token_id = tokenizer.convert_tokens_to_ids(TOKEN)
    if token_id != old_vocab or tokenizer_identity(tokenizer, exclude_agent=True) != before:
        raise ValueError('Old token IDs changed')
    model.llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        model.llm.get_input_embeddings().weight[token_id].copy_(input_mean)
        model.llm.get_output_embeddings().weight[token_id].copy_(output_mean)
    model.llm.requires_grad_(False)
    model.config.vocab_size = len(tokenizer)
    return {'protocol': PROTOCOL, 'agent_call_schema': 'task-start', 'token': TOKEN, 'token_id': token_id, 'old_vocab_size': old_vocab,
            'vocab_size': len(tokenizer), 'old_tokenizer_sha256': before,
            'tokenizer_sha256': tokenizer_identity(tokenizer), 'base_identity': base_identity(base_path)}


class RowEmbedding(nn.Module):
    def __init__(self, base, token_id):
        super().__init__()
        self.base = base
        self.token_id = token_id
        self.row = nn.Parameter(base.weight[token_id].detach().float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids):
        output = self.base(ids)
        return torch.where((ids == self.token_id).unsqueeze(-1), self.row.to(output.dtype), output)


class RowHead(nn.Module):
    def __init__(self, base, token_id):
        super().__init__()
        self.base = base
        self.token_id = token_id
        self.row = nn.Parameter(base.weight[token_id].detach().float().clone())

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        logits = self.base(hidden)
        new = torch.nn.functional.linear(hidden, self.row.to(hidden.dtype).unsqueeze(0))
        return torch.cat([logits[..., :self.token_id], new, logits[..., self.token_id + 1:]], -1)


def install_rows(llm, metadata):
    base = llm.get_base_model()
    base.set_input_embeddings(RowEmbedding(base.get_input_embeddings(), metadata['token_id']))
    base.set_output_embeddings(RowHead(base.get_output_embeddings(), metadata['token_id']))


def save_adapter(llm, tokenizer, metadata, path):
    from safetensors.torch import save_file
    path = Path(path)
    llm.save_pretrained(path, save_embedding_layers=False)
    base = llm.get_base_model()
    save_file({'input_row': base.get_input_embeddings().row.detach().cpu().contiguous(),
               'output_row': base.get_output_embeddings().row.detach().cpu().contiguous()},
              str(path / 'agent_rows.safetensors'))
    tokenizer.save_pretrained(path / 'tokenizer')
    (path / 'agent_protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    adapter_identity(path)


def load_adapter(model, base_path, path):
    from transformers import AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    path = Path(path)
    adapter_identity(path)
    meta = json.loads((path / 'agent_protocol.json').read_text())
    old = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path / 'tokenizer'), trust_remote_code=True)
    if (meta['base_identity'] != base_identity(base_path)
            or tokenizer_identity(old) != meta['old_tokenizer_sha256']
            or tokenizer_identity(tokenizer) != meta['tokenizer_sha256']
            or tokenizer_identity(tokenizer, exclude_agent=True) != tokenizer_identity(old)
            or len(tokenizer) != meta['vocab_size'] or len(old) != meta['old_vocab_size']
            or tokenizer.encode(TOKEN, add_special_tokens=False) != [meta['token_id']]
            or meta['token_id'] != len(old)):
        raise ValueError('Adapter base/tokenizer identity mismatch')
    rows = load_file(str(path / 'agent_rows.safetensors'))
    dimension = model.llm.get_input_embeddings().weight.shape[1]
    if set(rows) != {'input_row', 'output_row'} or any(row.shape != (dimension,) or not torch.isfinite(row).all() for row in rows.values()):
        raise ValueError('Invalid Agent token rows')
    model.llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.vocab_size = len(tokenizer)
    with torch.no_grad():
        model.llm.get_input_embeddings().weight[meta['token_id']].copy_(rows['input_row'])
        model.llm.get_output_embeddings().weight[meta['token_id']].copy_(rows['output_row'])
    model.llm = PeftModel.from_pretrained(model.llm, str(path)).merge_and_unload()
    model.prepare_processor(tokenizer=tokenizer)
    return meta
