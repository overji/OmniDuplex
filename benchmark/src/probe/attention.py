"""Bound native eager attention memory without dropping context or changing masks."""
from types import MethodType


def chunked_forward(original, chunk_size):
    if chunk_size < 1:
        raise ValueError('Attention query chunk size must be positive')

    def forward(module, query, key, value, *args, use_flash=True, **kwargs):
        length = query.shape[-2]
        if use_flash or length <= chunk_size:
            return original(query, key, value, *args, use_flash=use_flash, **kwargs)
        import torch
        mask = kwargs.get('attn_mask')
        if kwargs.get('is_causal', False):
            raise ValueError('Chunked attention requires the explicit native causal mask')
        outputs = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            part = dict(kwargs)
            if mask is not None and mask.shape[-2] != 1:
                part['attn_mask'] = mask[..., start:end, :]
            outputs.append(original(query[..., start:end, :], key, value,
                                    *args, use_flash=False, **part))
        # Baichuan LocalAttention returns [batch, query length, heads, head dim].
        return torch.cat(outputs, dim=1)
    return forward


def install_baichuan_attention(model, chunk_size):
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'LocalAttention':
            module.forward = MethodType(chunked_forward(module.forward, chunk_size), module)
            count += 1
    if not count:
        raise ValueError('Native Baichuan LocalAttention modules were not found')
    print(f'Baichuan eager attention: query chunks of {chunk_size}, full keys/values retained ({count} modules)', flush=True)
