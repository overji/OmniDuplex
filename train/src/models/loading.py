"""加载MiniCPM原生双工模型。task_start使用已有文本词表，禁用TTS；不管理训练步骤。"""

from pathlib import Path
from train.src.infrastructure.config import model_alias


def load_model(config, device):
    """返回(model, duplex)；保持BF16/无TTS配置，并在创建duplex前校验原始词表。这里只加载适配，不创建优化器或执行训练步。"""
    from train.src.domain.agent import require_protocol
    require_protocol(config)
    import torch
    from transformers import AutoModel
    path = Path(config['model_path']).resolve()
    # Remote-code loading treats dots in local directory names as module separators.
    alias = model_alias(config)
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        alias.symlink_to(path, target_is_directory=True)
    if alias.resolve() != path:
        raise ValueError('Model alias points to a different checkpoint')
    model = AutoModel.from_pretrained(str(alias), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation=config['attention_backend'], init_tts=False, low_cpu_mem_usage=True).eval().to(device)
    model.requires_grad_(False)
    model.init_tts = lambda **kwargs: None
    from transformers import AutoTokenizer
    from train.src.infrastructure.checkpoints import tokenizer_metadata, full_model_identity
    tokenizer = AutoTokenizer.from_pretrained(str(alias), trust_remote_code=True, local_files_only=True)
    from train.src.infrastructure.tokens import validate_text_separator, validate_vocabulary, validate_text_duplex
    validate_text_separator(tokenizer)
    tokenizer_metadata(tokenizer)
    validate_vocabulary(model, tokenizer)
    if (path / 'full_training_model.json').is_file():
        full_model_identity(path)
    model.prepare_processor(tokenizer=tokenizer)
    duplex = model.as_duplex(generate_audio=False, sliding_window_mode='off')
    validate_text_duplex(duplex)
    return model, duplex


def enable_full_training(model):
    """解冻MiniCPM的LLM/媒体/连接层，禁用TTS和有状态APM的不安全checkpoint重算。"""
    model.requires_grad_(True)
    if getattr(model,'tts',None) is not None:model.tts.requires_grad_(False)
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.llm.config.use_cache=False
    if hasattr(model.apm,'gradient_checkpointing_disable'):model.apm.gradient_checkpointing_disable()
    if hasattr(model.vpm,'gradient_checkpointing_enable'):model.vpm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
