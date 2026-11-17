"""Checkpoint immutable KV boundaries while retaining native decoder feed order."""
import torch
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten


def checkpoint_flat(function, *args, **kwargs):
    """Expose nested tensor inputs to checkpoint's saved-tensor/offload hooks.

    PyTorch 2.6 saves only top-level tensor arguments through save_for_backward.
    Capturing tuples of KV tensors as opaque arguments retains them on the GPU.
    The closure below holds only the callable and structural metadata; actual
    tensor leaves are independent checkpoint arguments, preserving their grads.
    """
    leaves, structure = tree_flatten((args, kwargs))

    def forward(*values):
        positional, named = tree_unflatten(list(values), structure)
        return function(*positional, **named)

    return checkpoint(forward, *leaves, use_reentrant=False)


@contextmanager
def checkpoint_vision_embedding(model, enabled=False):
    """Recompute stateless vision encoding while preserving native eval semantics.

    Native GRPO replay must stay in eval mode, where the vision encoder's
    training-only checkpoint switch is inactive. Checkpoint the complete vision
    call instead; never recompute the stateful streaming audio encoder. The
    captured bound method remains valid after this temporary override is removed.
    """
    if not enabled or not torch.is_grad_enabled():
        yield
        return
    original = model.get_vision_embedding
    had_override = 'get_vision_embedding' in model.__dict__

    def encode(data):
        return checkpoint_flat(original, dict(data))

    model.get_vision_embedding = encode
    try:
        yield
    finally:
        if had_override:
            model.get_vision_embedding = original
        else:
            delattr(model, 'get_vision_embedding')


def cached_hidden(llm, segments, checkpoint_segments=0, *, checkpoint_feeds=False):
    """Recompute groups of feeds on backward, without detaching their shared KV.

    DynamicCache mutates in place. Checkpoint inputs/outputs must instead be
    immutable legacy tuples; each invocation constructs its own cache wrapper.
    Every native feed and position stays identical to uncheckpointed replay.
    """
    from transformers.cache_utils import DynamicCache
    past, offset, states = None, 0, []
    size = checkpoint_segments or len(segments)
    for start in range(0, len(segments), size):
        feeds = tuple(segments[start:start + size])

        def forward(inputs, previous, position):
            cache = DynamicCache.from_legacy_cache(previous) if previous is not None else None
            hidden = []
            for segment in inputs:
                pos = torch.arange(position, position + len(segment), device=segment.device)[None]
                if checkpoint_feeds and torch.is_grad_enabled():
                    # Outer groups bound the number of saved KV boundaries.
                    # Inner checkpoints bound decoder activations during outer
                    # recomputation; otherwise every feed in that group retains
                    # its full decoder graph at once on the GPU.
                    previous = cache.to_legacy_cache() if cache is not None else None

                    def decode(feed, past, positions):
                        current = DynamicCache.from_legacy_cache(past) if past is not None else None
                        out = llm.model(inputs_embeds=feed[None], position_ids=positions,
                            past_key_values=current, use_cache=True, return_dict=True)
                        return out.last_hidden_state[0], out.past_key_values.to_legacy_cache()

                    state, next_past = checkpoint_flat(decode, segment, previous, pos)
                    cache = DynamicCache.from_legacy_cache(next_past)
                else:
                    out = llm.model(inputs_embeds=segment[None], position_ids=pos,
                        past_key_values=cache, use_cache=True, return_dict=True)
                    cache = out.past_key_values
                    state = out.last_hidden_state[0]
                hidden.append(state)
                position += len(segment)
            return torch.cat(hidden), cache.to_legacy_cache()

        if checkpoint_segments and torch.is_grad_enabled():
            hidden, past = checkpoint_flat(forward, feeds, past, offset)
        else:
            hidden, past = forward(feeds, past, offset)
        states.append(hidden)
        offset += sum(len(segment) for segment in feeds)
    return torch.cat(states)
