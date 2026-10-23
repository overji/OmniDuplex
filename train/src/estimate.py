"""Checkpoint-header parameter census; activation figures are planning estimates."""
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


def estimate(config, lengths):
    model_dir = Path(config['model_path'])
    counts = defaultdict(int)
    lora = 0
    for path in model_dir.glob('*.safetensors'):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        for name, value in header.items():
            if name == '__metadata__':
                continue
            shape = value['shape']
            group = 'tts' if name.startswith('tts.') else ('llm' if name.startswith('llm.') else 'encoders')
            counts[group] += math.prod(shape)
            if name.startswith('llm.') and len(shape) == 2 and any(name.endswith('.' + t + '.weight') for t in config['target_modules']):
                lora += config['lora_rank'] * sum(shape)
    if not counts:
        raise ValueError('No safetensors weights found')
    architecture = json.loads((model_dir / 'config.json').read_text())
    h, layers = architecture['hidden_size'], architecture['num_hidden_layers']
    gib = 2**30
    base = 2 * (counts['llm'] + counts['encoders']) / gib
    if config.get('training_method', 'lora_sft') != 'lora_sft':
        parameters = counts['llm'] + counts['encoders'] + 2 * h
        return {'parameter_counts': dict(counts), 'trainable_parameters': parameters,
            'bf16_model_gib': 2 * parameters / gib,
            'bf16_gradients_gib': 2 * parameters / gib,
            'fp32_master_and_adam_gib': 12 * parameters / gib,
            'total_unsharded_state_gib': 16 * parameters / gib,
            'assumptions': 'All LLM/vision/audio/connector parameters; no TTS. ZeRO-3 partitions GPU weights/gradients and CPU-offloaded FP32 masters/Adam by world size. Activations, communication buffers, native rollout KV and checkpoints are additional; CPU-offload uses substantial RAM. GRPO beta=0 uses no reference model. Not a measured peak.'}
    # PEFT adapters use fp32: weights, grads and two fp32 Adam states.
    token_parameters = 2 * h if config.get('agent_output_protocol') == 'agent_token_task_start_v1' else 0
    adapter = 16 * (lora + token_parameters) / gib
    rows = []
    for length in lengths:
        checkpoints = 2 * length * h * layers / gib
        # Checkpointed SDPA: conservative range for layer temporaries, workspaces and fragmentation.
        low = base + adapter + checkpoints * 2 + 3
        high = base + adapter + checkpoints * 5 + 7
        rows.append({'sequence_tokens': length, 'estimated_training_gib_low': round(low, 2),
                     'estimated_training_gib_high': round(high, 2)})
    return {'parameter_counts': dict(counts), 'lora_parameters': lora, 'new_token_parameters': token_parameters,
        'frozen_bf16_weights_gib_excluding_tts': base, 'fp32_lora_grad_adam_gib': adapter,
        'estimates': rows, 'assumptions': 'batch=1, BF16 frozen weights, FP32 LoRA, gradient checkpointing, configured memory-efficient attention, target-only chunked logits; no KV cache. Activation range is heuristic, not measured. Encoding peak and CUDA context may differ.'}
