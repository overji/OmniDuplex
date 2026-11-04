"""Checkpoint immutable KV boundaries while retaining native decoder feed order."""
import torch
from torch.utils.checkpoint import checkpoint


def cached_hidden(llm, segments, checkpoint_segments=0):
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
                out = llm.model(inputs_embeds=segment[None], position_ids=pos,
                    past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                hidden.append(out.last_hidden_state[0])
                position += len(segment)
            return torch.cat(hidden), cache.to_legacy_cache()

        if checkpoint_segments and torch.is_grad_enabled():
            hidden, past = checkpoint(forward, feeds, past, offset, use_reentrant=False)
        else:
            hidden, past = forward(feeds, past, offset)
        states.append(hidden)
        offset += sum(len(segment) for segment in feeds)
    return torch.cat(states)
