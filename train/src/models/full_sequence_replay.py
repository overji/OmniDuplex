"""Causal trajectory replay without differentiable histories of KV snapshots."""
from contextlib import contextmanager, ExitStack

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def checkpoint_decoder_layers(decoder):
    """Checkpoint eval-mode layers without enabling dropout or changing audio state.

    Bound methods are captured independently per layer. Backward recomputation
    may happen after the temporary overrides have been removed.
    """
    if not torch.is_grad_enabled():
        yield
        return
    def wrap(original):
        def forward(*args, **kwargs):
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        return forward
    def restore(layer, original, overridden):
        if overridden:
            layer.forward = original
        else:
            delattr(layer, 'forward')
    with ExitStack() as stack:
        for layer in decoder.layers:
            original = layer.forward
            overridden = 'forward' in layer.__dict__
            layer.forward = wrap(original)
            stack.callback(restore, layer, original, overridden)
        yield


def full_sequence_hidden(llm, embeddings):
    """Use all observed/generated inputs with causal attention and no KV cache."""
    if llm.training:
        raise ValueError('Full-sequence GRPO replay requires eval mode')
    positions = torch.arange(len(embeddings), device=embeddings.device)[None]
    with checkpoint_decoder_layers(llm.model):
        return llm.model(inputs_embeds=embeddings[None], position_ids=positions,
                         past_key_values=None, use_cache=False,
                         return_dict=True).last_hidden_state[0]
