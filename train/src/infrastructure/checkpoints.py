"""checkpoint身份、保存与导出。保留数据/配置/tokenizer指纹及优化器游标契约。"""

import hashlib
import json
from pathlib import Path

from benchmark.data.agent_token import base_identity, tokenizer_identity
from benchmark.data.files import write_json
from train.src.domain.agent import PROTOCOL, SEPARATOR


def tokenizer_metadata(tokenizer):
    from train.src.infrastructure.tokens import validate_task_token
    token_id = validate_task_token(tokenizer)
    ids = [token_id]
    if '<|agent|>' in tokenizer.get_vocab():
        raise ValueError('Legacy Agent token is incompatible with atomic task_start SFT')
    return {'protocol': PROTOCOL, 'agent_call_schema': 'task-start', 'separator': SEPARATOR,
            'separator_token_ids': ids, 'separator_is_special': True, 'vocab_size': len(tokenizer),
            'tokenizer_sha256': tokenizer_identity(tokenizer)}








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


def fingerprints(prepared, config):
    """计算prepared、提示词、ZeRO配置和基座身份，用于恢复时拒绝不一致输入；不修改这些文件。"""
    files = {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest()
             for name in ('train.json', 'validation.json', 'manifest.json')}
    for key in ('system_prompt', 'agent_prompt', 'deepspeed_config'):
        files[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    from benchmark.data.agent_token import base_identity
    files['base_model'] = base_identity(config['model_path'])
    from train.src.infrastructure.tokens import load_training_tokenizer
    files['tokenizer'] = tokenizer_metadata(load_training_tokenizer(config))
    return files


def save(engine, path, epoch, cursor, identity, config, duplex, rank):
    """所有rank共同保存ZeRO状态及各自RNG；仅rank0写tokenizer/config。epoch/cursor必须指向已完成的训练位置。"""
    import torch
    from train.src.infrastructure.distributed import rng_state
    path.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(str(path), tag='state', client_state=dict(epoch=epoch, cursor=cursor,
        identity=identity, config=config))
    torch.save(rng_state(), path / f'rng_rank{rank}.pt')
    if rank == 0:
        duplex.tokenizer.save_pretrained(path / 'tokenizer')
        write_json(path / 'training_config.json', config)
    torch.distributed.barrier()


def export_checkpoint(checkpoint, output):
    """在CPU合并ZeRO权重并导出HF模型，不改变tokenizer/词表；主存及导出目录由调用方保证。"""
    import torch
    from transformers import AutoModel, AutoTokenizer
    if output.exists():
        raise ValueError('Use a fresh export directory')
    config = json.loads((checkpoint / 'training_config.json').read_text())
    from train.src.domain.agent import require_protocol
    require_protocol(config)
    # Load on CPU; exporting an optimizer checkpoint needs substantial host RAM.
    from train.src.infrastructure.config import model_alias
    alias = model_alias(config)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(Path(config['model_path']).resolve(), target_is_directory=True)
    model = AutoModel.from_pretrained(str(alias), trust_remote_code=True, torch_dtype=torch.bfloat16,
                                     init_tts=False, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / 'tokenizer'), trust_remote_code=True)
    from train.src.infrastructure.tokens import resize_vocabulary
    tokenizer_metadata(tokenizer)
    resize_vocabulary(model, tokenizer)
    state = read_zero_state(checkpoint)
    state = {key.removeprefix('model.'): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size='5GB')
    tokenizer.save_pretrained(output)
    write_json(output / 'training_config.json', config)
    finalize_export(output, checkpoint, config, tokenizer)


def finalize_export(output, checkpoint, config, tokenizer):
    import uuid
    source = Path(config['model_path']).resolve()
    runtime_hashes = copy_runtime_files(source, output)
    from train.src.domain.agent import require_protocol
    require_protocol(config)
    write_json(output / 'full_training_model.json', dict(**tokenizer_metadata(tokenizer),
        export_id=str(uuid.uuid4()),
        files=base_identity(output), runtime_files=runtime_hashes, source_checkpoint=str(checkpoint.resolve())))


def copy_runtime_files(source, output):
    """导出MiniCPM时补齐原生运行时代码和媒体processor配置，不复制基座tokenizer。"""
    import shutil
    # Remote AutoModel saving does not reliably copy the outer MiniCPMO class
    # after nested models register their own auto classes. Include its local
    # runtime code and media processor settings explicitly, never base tokenizer.
    runtime_files = list(source.glob('*.py'))
    runtime_files += [source / name for name in ('preprocessor_config.json', 'processor_config.json')
                      if (source / name).is_file()]
    runtime_hashes = {}
    for file in runtime_files:
        shutil.copy2(file, output / file.name)
        runtime_hashes[file.name] = hashlib.sha256((output / file.name).read_bytes()).hexdigest()
    return runtime_hashes


def read_zero_state(checkpoint):
    """仅在显式导出时于CPU合并分片；普通身份检查不导入DeepSpeed。"""
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    return get_fp32_state_dict_from_zero_checkpoint(str(checkpoint), tag='state')
