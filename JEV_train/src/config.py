import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'jev_gate_qwen25_full_sft_token_v3'
SILENT, RESPONSE = '<|silent|>', '<|response|>'
AGENT = '<|task_start|>'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    config = json.loads(Path(path).read_text())
    if any(k in config for k in ('lora_rank', 'lora_alpha', 'lora_dropout', 'target_modules', 'classifier_adapter', 'backend_adapter')):
        raise ValueError('Legacy LoRA configuration is unsupported; use the full-SFT checkpoint configuration')
    if config.get('training_method') != 'full_sft' or not config.get('deepspeed_config'):
        raise ValueError('JEV supports full_sft with a DeepSpeed configuration only')
    if config['dtype'] not in ('bfloat16','float32'):
        raise ValueError('Full SFT supports bfloat16 or float32')
    if config['mode'] not in ('classifier_only', 'separate_sft') or config['backend'] != 'qwen25_omni':
        raise ValueError('Supported modes: classifier_only/separate_sft; backend: qwen25_omni')
    for key in ('data', 'classifier_model', 'backend_model', 'videos_dir', 'cache_dir', 'classifier_checkpoint', 'backend_checkpoint', 'deepspeed_config'):
        if config.get(key):
            p = Path(config[key]).expanduser()
            config[key] = str(p if p.is_absolute() else ROOT / p)
    if not 0 < config['threshold'] < 1:
        raise ValueError('threshold must be between zero and one')
    if not 0 < config['validation_fraction'] + config['test_fraction'] < 1 or min(config['validation_fraction'], config['test_fraction']) <= 0:
        raise ValueError('Need nonempty train/validation/test proportions')
    for key in ('max_prefix_seconds', 'max_sequence_tokens', 'negative_ratio', 'epochs', 'gradient_accumulation_steps', 'max_new_tokens', 'max_pixels'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'max_grad_norm', 'cache_atol', 'cache_rtol'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    return config


def write(path, value):
    from benchmark.data.files import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def base_identity(path):
    path=Path(path)
    return {name:hashlib.sha256((path/name).read_bytes()).hexdigest()
            for name in ('config.json','model.safetensors.index.json','tokenizer_config.json') if (path/name).is_file()}
