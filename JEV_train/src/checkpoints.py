"""JEV完整模型导出/身份校验。训练保存ZeRO状态，显式export在CPU还原HF模型。"""
import hashlib
import json
from pathlib import Path
import uuid

from .config import PROTOCOL, digest, base_identity, write


def weight_files(path):
    index=path/'model.safetensors.index.json'
    return set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}


def validate_gate_export(path,config):
    meta=json.loads((path/'jev_gate.json').read_text())
    if meta.get('protocol')!=PROTOCOL or meta.get('training_method')!='full_sft' or meta.get('base')!=str(Path(config['classifier_model']).resolve()):
        raise ValueError('Classifier full-SFT protocol/base mismatch')
    if meta['base_identity']!=base_identity(config['classifier_model']) or meta['files']!=base_identity(path):
        raise ValueError('Classifier full-SFT identity mismatch')
    if not all((path/name).is_file() for name in weight_files(path)):
        raise ValueError('Incomplete classifier model weights')
    head=path/'jev_gate_head.safetensors'
    if not head.is_file() or hashlib.sha256(head.read_bytes()).hexdigest()!=meta['head_sha256']:
        raise ValueError('Classifier decision head changed')
    if any(not (path/name).is_file() or hashlib.sha256((path/name).read_bytes()).hexdigest()!=value for name,value in meta.get('runtime_files',{}).items()):
        raise ValueError('Classifier runtime files changed')
    return meta


def export_checkpoint(checkpoint,output):
    import torch
    from transformers import AutoModel,AutoTokenizer,Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
    from train.src.infrastructure.tokens import resize_vocabulary,validate_text_separator
    from train.src.infrastructure.checkpoints import copy_runtime_files,read_zero_state
    if output.exists():raise ValueError('Use a fresh export directory')
    config=json.loads((checkpoint/'training_config.json').read_text())
    if config.get('protocol')!=PROTOCOL or config.get('training_method')!='full_sft':
        raise ValueError('Only current JEV full-SFT checkpoints can be exported')
    stage=config['jev_stage'];tokenizer=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer'),trust_remote_code=True,local_files_only=True)
    state=read_zero_state(checkpoint)
    weights={key.removeprefix('model.'):value for key,value in state.items() if key.startswith('model.')}
    if stage=='classifier':
        from safetensors.torch import save_file
        source=Path(config['classifier_model']).resolve()
        alias=Path(config['cache_dir'])/('gate_base_'+digest(str(source))[:12])
        alias.parent.mkdir(parents=True,exist_ok=True)
        if not alias.exists():alias.symlink_to(source,target_is_directory=True)
        if alias.resolve()!=source:raise ValueError('Classifier export alias changed')
        model=AutoModel.from_pretrained(str(alias),trust_remote_code=True,torch_dtype=torch.bfloat16,init_tts=False,low_cpu_mem_usage=True)
        validate_text_separator(tokenizer);resize_vocabulary(model,tokenizer);model.load_state_dict(weights,strict=True)
        model.save_pretrained(output,safe_serialization=True,max_shard_size='5GB');tokenizer.save_pretrained(output)
        runtime=copy_runtime_files(source,output)
        head=state['head.weight'].detach().float().cpu().contiguous()
        if head.shape!=(2,model.llm.config.hidden_size) or not torch.isfinite(head).all():raise ValueError('Invalid classifier head')
        save_file({'weight':head},str(output/'jev_gate_head.safetensors'))
        write(output/'jev_gate.json',dict(protocol=PROTOCOL,training_method='full_sft',base=str(source),base_identity=base_identity(source),
              token_ids=[tokenizer.convert_tokens_to_ids(t) for t in ('<|silent|>','<|response|>')],
              task_start_ids=validate_text_separator(tokenizer),tokenizer_sha256=digest(tokenizer.get_vocab()),files=base_identity(output),
              runtime_files=runtime,head_sha256=hashlib.sha256((output/'jev_gate_head.safetensors').read_bytes()).hexdigest(),export_id=str(uuid.uuid4())))
    elif stage=='backend':
        model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(config['backend_model'],torch_dtype=torch.bfloat16,local_files_only=True)
        model.load_state_dict(weights,strict=True);model.save_pretrained(output,safe_serialization=True,max_shard_size='5GB')
        processor=Qwen2_5OmniProcessor.from_pretrained(config.get('backend_checkpoint') or config['backend_model'],local_files_only=True,
                                                   use_fast=config.get('processor_use_fast',False))
        if digest(tokenizer.get_vocab())!=digest(processor.tokenizer.get_vocab()):raise ValueError('Backend export tokenizer mismatch')
        processor.save_pretrained(output);tokenizer.save_pretrained(output)
        write(output/'jev_backend.json',dict(protocol=PROTOCOL,training_method='full_sft',base=config['backend_model'],
              base_identity=base_identity(config['backend_model']),files=base_identity(output),tokenizer=digest(tokenizer.get_vocab()),export_id=str(uuid.uuid4())))
    else:raise ValueError('Unknown JEV checkpoint stage')
    write(output/'training_config.json',config)
