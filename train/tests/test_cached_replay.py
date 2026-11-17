"""A tiny real causal decoder checks that checkpointed KV keeps cross-feed gradients."""
import copy
import torch
import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM
from train.src.models.cached_replay import cached_hidden


def test_flat_checkpoint_exposes_nested_tensors_to_offload_and_releases_original():
    import gc
    import weakref
    from train.src.models.cached_replay import checkpoint_flat
    tensor = torch.randn(7)
    expected = tensor.clone()
    reference = weakref.ref(tensor)
    parameter = torch.nn.Parameter(torch.ones(7))
    observed = []
    def pack(value):
        observed.append(tuple(value.shape))
        return value.detach().clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value:value):
        output = checkpoint_flat(lambda data: (parameter*data['values'][0]).sin(),
                                 {'values':[tensor]})
    del tensor
    gc.collect()
    assert (7,) in observed
    assert reference() is None  # No opaque dict/tuple keeps the old tensor alive.
    output.sum().backward()
    torch.testing.assert_close(parameter.grad, expected*expected.cos())


def test_kv_boundaries_reach_saved_tensor_hooks_without_changing_gradients():
    torch.manual_seed(91)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    baseline = Qwen2ForCausalLM(config).eval()
    checked = copy.deepcopy(baseline)
    feeds = [torch.randn(n,16,requires_grad=True) for n in (3,1,2,1,2)]
    clones = [x.detach().clone().requires_grad_() for x in feeds]
    captured = []
    def pack(value):
        if value.ndim == 4:
            captured.append(tuple(value.shape))
        return value.detach().clone()
    reference = cached_hidden(baseline, feeds)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value:value):
        actual = cached_hidden(checked, clones, 2)
        actual[-1].square().sum().backward()
    assert len(captured) >= 4  # Keys/values for both decoder layers at boundaries.
    assert any(shape[-2] == 4 for shape in captured)
    torch.testing.assert_close(actual, reference)
    reference[-1].square().sum().backward()
    for a,b in zip(feeds,clones):
        torch.testing.assert_close(a.grad,b.grad)
    for a,b in zip(baseline.parameters(),checked.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad)


@pytest.mark.parametrize('checkpoint_feeds', [False, True])
def test_cached_checkpoint_matches_native_values_and_gradients(checkpoint_feeds):
    torch.manual_seed(73)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        attention_dropout=0.)
    first = Qwen2ForCausalLM(config).eval()
    other = copy.deepcopy(first)
    feeds = [torch.randn(n, 16, requires_grad=True) for n in (3, 1, 2, 1, 1)]
    clones = [v.detach().clone().requires_grad_() for v in feeds]
    native = cached_hidden(first, feeds)
    checked = cached_hidden(other, clones, 2, checkpoint_feeds=checkpoint_feeds)
    torch.testing.assert_close(native, checked)
    # Loss only on the last feed still propagates through earlier KV boundaries.
    native[-1].square().sum().backward()
    checked[-1].square().sum().backward()
    for a, b in zip(feeds, clones):
        torch.testing.assert_close(a.grad, b.grad)
    for a, b in zip(first.parameters(), other.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_eval_vision_checkpoint_preserves_values_gradients_and_method():
    from train.src.models.cached_replay import checkpoint_vision_embedding

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(),
                                              torch.nn.Linear(16, 4))
            self.calls = 0

        def get_vision_embedding(self, data):
            self.calls += 1
            assert not self.training
            return [self.layers(data['pixels'][0])]

    torch.manual_seed(37)
    first = Vision().eval()
    other = copy.deepcopy(first)
    data = {'pixels': [torch.randn(2, 8)]}  # Real media inputs need no gradients.
    reference = first.get_vision_embedding(data)[0]
    with checkpoint_vision_embedding(other, True):
        actual = other.get_vision_embedding(data)[0]
    assert 'get_vision_embedding' not in other.__dict__
    torch.testing.assert_close(actual, reference)
    reference.square().sum().backward()
    actual.square().sum().backward()  # Recompute after override removal.
    assert other.calls == 2
    for a, b in zip(first.parameters(), other.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert not other.training
    original = other.get_vision_embedding
    with pytest.raises(RuntimeError, match='test restore'):
        with checkpoint_vision_embedding(other, True):
            raise RuntimeError('test restore')
    assert other.get_vision_embedding == original
    with torch.no_grad(), checkpoint_vision_embedding(other, True):
        assert other.get_vision_embedding == original


def test_vision_checkpoint_restores_existing_instance_override():
    from train.src.models.cached_replay import checkpoint_vision_embedding
    model = torch.nn.Module()
    original = lambda data: [data['pixels'].square()]
    model.get_vision_embedding = original
    with checkpoint_vision_embedding(model, True):
        assert model.get_vision_embedding is not original
    assert model.get_vision_embedding is original
