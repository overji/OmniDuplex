"""文本分隔符和词表校验；保留历史导出读取及JEV分类token扩展工具。"""
from train.src.domain.agent import SEPARATOR, validate_text_separator


def load_training_tokenizer(config):
    from pathlib import Path
    from transformers import AutoTokenizer
    from train.src.infrastructure.config import model_alias
    alias = model_alias(config)
    path = alias if alias.exists() and alias.resolve() == Path(config['model_path']).resolve() else Path(config['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    validate_text_separator(tokenizer)
    return tokenizer


def validate_task_token(tokenizer):
    ids = tokenizer.encode(SEPARATOR, add_special_tokens=False)
    if len(ids) != 1 or ids[0] not in tokenizer.all_special_ids:
        raise ValueError('task_start must be one registered special token; regenerate tokenizer/prepared data')
    if tokenizer.get_vocab().get(SEPARATOR) != ids[0]:
        raise ValueError('task_start token identity mismatch')
    return ids[0]


def register_task_token(tokenizer):
    """只在内存中追加一次；已有原子token保持ID，旧词表的每个ID必须原样保留。"""
    before = tokenizer.get_vocab().copy()
    if SEPARATOR not in before:
        old_size = len(tokenizer)
        added = tokenizer.add_special_tokens({'additional_special_tokens': [SEPARATOR]},
                                             replace_additional_special_tokens=False)
        after = tokenizer.get_vocab()
        if added != 1 or after[SEPARATOR] != old_size or len(tokenizer) != old_size + 1:
            raise ValueError('task_start must append exactly one vocabulary row')
        if any(after.get(token) != index for token, index in before.items()):
            raise ValueError('Registering task_start changed an existing token ID')
    return validate_task_token(tokenizer)


def mean_row(weight):
    import torch
    total = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
    for start in range(0, len(weight), 2048):
        total += weight[start:start + 2048].float().sum(0)
    return total / len(weight)


def resize_vocabulary(model, tokenizer):
    """同步输入/输出表和config；仅初始化新行。相同词表重复加载时绝不重置已训练权重。"""
    import torch
    llm = getattr(model, 'llm', model)
    old_size = llm.get_input_embeddings().weight.shape[0]
    if llm.get_output_embeddings().weight.shape[0] != old_size or old_size > len(tokenizer):
        raise ValueError('Tokenizer/input/output vocabulary mismatch')
    if old_size < len(tokenizer):
        input_mean = mean_row(llm.get_input_embeddings().weight).detach()
        output_mean = mean_row(llm.get_output_embeddings().weight).detach()
        llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        with torch.no_grad():
            llm.get_input_embeddings().weight[old_size:].copy_(input_mean)
            llm.get_output_embeddings().weight[old_size:].copy_(output_mean)
    llm.config.vocab_size = len(tokenizer)
    model.config.vocab_size = len(tokenizer)


def validate_duplex_token(duplex):
    token = validate_task_token(duplex.tokenizer)
    forbidden = set(getattr(duplex, 'forbidden_token_ids', ()))
    forbidden.update(getattr(duplex.decoder, 'forbidden_token_ids', ()))
    for name in ('listen_token_id', 'speak_token_id', 'tts_bos_token_id',
                 'turn_eos_token_id', 'chunk_eos_token_id', 'chunk_tts_eos_token_id'):
        forbidden.add(getattr(duplex, name, None))
    if token in forbidden:
        raise ValueError('task_start conflicts with a native control or forbidden token')




def validate_vocabulary(model, tokenizer):
    """Text SFT cannot append or delete input/output vocabulary rows."""
    llm = getattr(model, 'llm', model)
    if any(table.weight.shape[0] != len(tokenizer) for table in
           (llm.get_input_embeddings(), llm.get_output_embeddings())):
        raise ValueError('Tokenizer/input/output vocabulary mismatch; text SFT never resizes vocabulary')


def validate_text_duplex(duplex):
    ids = validate_text_separator(duplex.tokenizer)
    forbidden = set(getattr(duplex, 'forbidden_token_ids', ()))
    forbidden.update(getattr(duplex.decoder, 'forbidden_token_ids', ()))
    if forbidden.intersection(ids):
        raise ValueError('Text separator contains forbidden native tokens')
