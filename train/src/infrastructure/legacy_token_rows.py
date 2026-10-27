"""仅供历史benchmark adapter推理，当前训练不支持LoRA。"""
import json
from pathlib import Path
import torch
from benchmark.data.agent_token import TOKEN, base_identity, tokenizer_identity, adapter_identity


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
