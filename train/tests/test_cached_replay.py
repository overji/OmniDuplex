"""A tiny real causal decoder checks that checkpointed KV keeps cross-feed gradients."""
import copy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from train.src.models.cached_replay import cached_hidden


def test_cached_checkpoint_matches_native_values_and_gradients():
    torch.manual_seed(73)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        attention_dropout=0.)
    first = Qwen2ForCausalLM(config).eval()
    other = copy.deepcopy(first)
    feeds = [torch.randn(n, 16, requires_grad=True) for n in (3, 1, 2, 1, 1)]
    clones = [v.detach().clone().requires_grad_() for v in feeds]
    native = cached_hidden(first, feeds)
    checked = cached_hidden(other, clones, 2)
    torch.testing.assert_close(native, checked)
    # Loss only on the last feed still propagates through earlier KV boundaries.
    native[-1].square().sum().backward()
    checked[-1].square().sum().backward()
    for a, b in zip(feeds, clones):
        torch.testing.assert_close(a.grad, b.grad)
    for a, b in zip(first.parameters(), other.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)
