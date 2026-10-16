"""Saved-tensor CPU offload with deduplicated parameter views for KV replay.

Native decoding reuses each weight many times. save_on_cpu alone copies that
weight once per use, which can exceed host RAM. Parameter identity and view
metadata permit one CPU copy per weight/view during an optimizer-free backward.
"""
from contextlib import contextmanager

import torch


def copy_strided(tensor, device):
    # xFormers attention biases use padded row strides. A plain .to('cpu')
    # compacts sliced views; restoring such a tensor breaks kernel alignment.
    result = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
    target, source = result, tensor.detach()
    for axis, (size, stride) in enumerate(zip(tensor.shape, tensor.stride())):
        if size > 1 and stride == 0:
            target = target.narrow(axis, 0, 1)
            source = source.narrow(axis, 0, 1)
    target.copy_(source)
    return result


@contextmanager
def saved_on_cpu(budget_gib=128):
    cache = {}
    stats = dict(saved_bytes=0, parameter_cache_hits=0)
    budget = budget_gib * 2**30
    def pack(tensor):
        base = tensor
        while getattr(base, '_base', None) is not None:
            base = base._base
        key = None
        if isinstance(base, torch.nn.Parameter):
            key = (id(base), base._version, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
        if key is not None and key in cache:
            stats['parameter_cache_hits'] += 1
            return tensor.device, cache[key]
        storage_elements = (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) if tensor.numel() else 0
        size = storage_elements * tensor.element_size()
        if budget and stats['saved_bytes'] + size > budget:
            raise RuntimeError('GRPO saved activations exceed grpo_activation_cpu_budget_gib; shorten the horizon or explicitly increase the CPU budget')
        value = copy_strided(tensor, 'cpu')
        stats['saved_bytes'] += size
        if key is not None:
            cache[key] = value
        return tensor.device, value
    def unpack(packed):
        device, tensor = packed
        return copy_strided(tensor, device)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats
