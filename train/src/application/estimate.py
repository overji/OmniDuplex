"""从safetensors头部估算状态显存；激活数值是预算，不是实测峰值。"""

import json
import math
import struct
from collections import defaultdict
from pathlib import Path


def estimate(config, lengths):
    model_dir = Path(config['model_path'])
    counts = defaultdict(int)
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
    if not counts:
        raise ValueError('No safetensors weights found')
    architecture = json.loads((model_dir / 'config.json').read_text())
    gib = 2**30
    # 新token为输入与输出各新增一行；已扩表的导出不再重复计数。
    token_file=model_dir/'tokenizer.json'
    token_data=json.loads(token_file.read_text()) if token_file.is_file() else {}
    from train.src.domain.agent import SEPARATOR
    has_marker=SEPARATOR in token_data.get('model',{}).get('vocab',{}) or any(t.get('content')==SEPARATOR for t in token_data.get('added_tokens',[]))
    new_parameters=(0 if has_marker else architecture['hidden_size']*(1 if architecture.get('tie_word_embeddings',False) else 2))
    parameters = counts['llm'] + counts['encoders'] + new_parameters
    return {'parameter_counts': dict(counts), 'trainable_parameters': parameters, 'new_token_parameters':new_parameters,
        'bf16_model_gib': 2 * parameters / gib,
        'bf16_gradients_gib': 2 * parameters / gib,
        'fp32_master_and_adam_gib': 12 * parameters / gib,
        'total_unsharded_state_gib': 16 * parameters / gib,
        'assumptions': 'All LLM/vision/audio/connector parameters; no TTS. ZeRO-3 partitions GPU weights/gradients and CPU-offloaded FP32 masters/Adam by world size. Activations, communication buffers, native rollout KV and checkpoints are additional; CPU-offload uses substantial RAM. Not a measured peak.'}
