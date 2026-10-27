"""配置与路径解析。ROOT固定为仓库根目录，不依赖运行命令时的工作目录。"""

import json
from pathlib import Path
from train.src.domain.agent import require_protocol

ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    """读取JSON并返回校验后的配置；相对路径按仓库ROOT解析，不写回原配置文件。"""
    config = json.loads(Path(path).read_text())
    method = config.setdefault('training_method', 'full_sft')
    if method not in ('full_sft', 'full_grpo'):
        raise ValueError('Unsupported training_method')
    if any(k in config for k in ('lora_rank','lora_alpha','lora_dropout','target_modules')):
        raise ValueError('LoRA configuration is unsupported; use a full-parameter configuration')
    require_protocol(config)
    paths = ['model_path', 'videos_dir', 'media_cache', 'system_prompt', 'agent_prompt']
    if config.get('data'):
        paths.append('data')
    for key in paths:
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else ROOT / p)
    if config.get('deepspeed_config'):
        p = Path(config['deepspeed_config'])
        config['deepspeed_config'] = str(p if p.is_absolute() else ROOT / p)
    for key in ('max_prefix_seconds', 'max_sequence_tokens',
                'gradient_accumulation_steps', 'epochs', 'loss_projection_chunk'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['max_speak_tokens_per_chunk'] < 4:
        raise ValueError('Chunk budget must fit speak, text, turn and chunk terminators')
    if 'sft_activation_offload' in config and type(config['sft_activation_offload']) is not bool:
        raise ValueError('sft_activation_offload must be a boolean')
    if config.get('sft_activation_offload', False):
        import math
        budget = config.get('sft_activation_cpu_budget_gib', 128)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError('sft_activation_cpu_budget_gib must be positive and finite')
    if method == 'full_grpo':
        if type(config.get('group_size', 4)) is not int or config.get('group_size', 4) < 2 or config.get('kl_beta', 0) != 0:
            raise ValueError('GRPO requires group_size >= 2 and kl_beta=0 in this implementation')
        if config.get('temperature', 1.) != 1.:
            raise ValueError('Native two-stage GRPO replay currently requires temperature=1')
        if not 0 < config.get('clip_epsilon', .2) < 1 or config.get('replay_logprob_tolerance', .15) <= 0:
            raise ValueError('Invalid GRPO clip or replay tolerance')
    if not config.get('deepspeed_config'):
        raise ValueError('Full training requires deepspeed_config')
    return config


def model_alias(config):
    """只计算无点号的本地模型别名路径，避免remote-code模块名解析问题；创建链接由调用方负责。"""
    import hashlib
    suffix = hashlib.sha256(str(Path(config['model_path']).resolve()).encode()).hexdigest()[:12]
    return Path(config['media_cache']) / 'training_models' / ('minicpmo_' + suffix)
