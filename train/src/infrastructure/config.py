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
    parallel = config.get('parallel_mode', 'replicated_qa')
    if parallel not in ('replicated_qa', 'data_parallel'):
        raise ValueError('Invalid parallel_mode')
    if parallel == 'data_parallel' and method != 'full_sft':
        raise ValueError('Independent-QA data parallel currently supports full_sft only')
    if type(config.get('save_total_limit', 0)) is not int or config.get('save_total_limit', 0) < 0:
        raise ValueError('save_total_limit must be a nonnegative integer')
    if type(config.get('length_bucket_size', 0)) is not int or config.get('length_bucket_size', 0) < 0:
        raise ValueError('length_bucket_size must be a nonnegative integer')
    gpu_budget = config.get('sft_activation_gpu_budget_gib', 0)
    if isinstance(gpu_budget, bool) or not isinstance(gpu_budget, (int, float)) or not 0 <= gpu_budget < float('inf'):
        raise ValueError('sft_activation_gpu_budget_gib must be finite and nonnegative')
    if 'wandb' in config:
        settings = config['wandb']
        if not isinstance(settings, dict) or set(settings) - {'project', 'entity', 'name', 'mode'}:
            raise ValueError('wandb accepts project/entity/name/mode only; use WANDB_API_KEY in the environment')
        if settings.get('mode', 'online') not in ('online', 'offline', 'disabled'):
            raise ValueError('Invalid wandb mode')
        if settings.get('mode') != 'disabled' and not settings.get('project'):
            raise ValueError('wandb requires a project')
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
    tail = config.get('post_response_silence_seconds', 0)
    if type(tail) is not int or tail < 0:
        raise ValueError('post_response_silence_seconds must be a nonnegative integer')
    if tail and (method != 'full_sft' or not config['supervise_silence']):
        raise ValueError('Post-response silence requires full_sft and supervise_silence=true')
    if config['max_speak_tokens_per_chunk'] < 4:
        raise ValueError('Chunk budget must fit speak, text, turn and chunk terminators')
    if 'sft_activation_offload' in config and type(config['sft_activation_offload']) is not bool:
        raise ValueError('sft_activation_offload must be a boolean')
    if config.get('sft_activation_offload', False):
        import math
        budget = config.get('sft_activation_cpu_budget_gib', 128)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError('sft_activation_cpu_budget_gib must be positive and finite')
    reserve = config.get('sft_activation_gpu_reserve_gib', 12)
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not 0 <= reserve < float('inf'):
        raise ValueError('sft_activation_gpu_reserve_gib must be finite and nonnegative')
    if method == 'full_grpo':
        if type(config.get('group_size', 4)) is not int or config.get('group_size', 4) < 2 or config.get('kl_beta', 0) != 0:
            raise ValueError('GRPO requires group_size >= 2 and kl_beta=0 in this implementation')
        if config.get('temperature', 1.) != 1.:
            raise ValueError('Native two-stage GRPO replay currently requires temperature=1')
        if not 0 < config.get('clip_epsilon', .2) < 1 or config.get('replay_logprob_tolerance', .15) <= 0:
            raise ValueError('Invalid GRPO clip or replay tolerance')
    backend = config.get('reward_backend', 'rules')
    if backend not in ('rules', 'llm'):
        raise ValueError('Unknown reward_backend')
    if backend == 'llm':
        import math
        from urllib.parse import urlparse
        if method != 'full_grpo':
            raise ValueError('LLM reward requires full_grpo')
        judge = config['judge']
        url = urlparse(judge['base_url'])
        if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password:
            raise ValueError('Training judge must use a localhost HTTP endpoint without URL credentials')
        if not isinstance(judge.get('model'), str) or not judge['model'].strip():
            raise ValueError('Judge requires an explicit served model name')
        for key in ('max_tokens', 'concurrency', 'timeout'):
            if type(judge[key]) is not int or judge[key] <= 0:
                raise ValueError(f'Invalid judge {key}')
        if type(judge['retries']) is not int or judge['retries'] < 0:
            raise ValueError('Invalid judge retries')
        for key in ('model_path', 'system_prompt', 'cache_dir'):
            path = Path(judge[key]).expanduser()
            judge[key] = str(path if path.is_absolute() else ROOT / path)
        if 'api_key' in judge:
            raise ValueError('Keep credentials in api_key_env, never in config')
        for key in ('timing', 'content', 'agent_timing', 'agent_content', 'agent_call', 'silence', 'invalid_call_penalty'):
            value = config['reward_weights'][key]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid reward weight: {key}')
    if not config.get('deepspeed_config'):
        raise ValueError('Full training requires deepspeed_config')
    if parallel == 'data_parallel':
        zero = json.loads(Path(config['deepspeed_config']).read_text())
        if zero['zero_optimization']['stage'] != 2:
            raise ValueError('Independent-QA mode requires ZeRO-2; conditional streaming forwards cannot use ZeRO-3')
    return config


def model_alias(config):
    """只计算无点号的本地模型别名路径，避免remote-code模块名解析问题；创建链接由调用方负责。"""
    import hashlib
    suffix = hashlib.sha256(str(Path(config['model_path']).resolve()).encode()).hexdigest()[:12]
    return Path(config['media_cache']) / 'training_models' / ('minicpmo_' + suffix)
