"""Cached CUDA memory must remain usable on every QA, without changing gradients."""
from unittest.mock import patch

import pytest
import torch

from train.src.infrastructure.offload import saved_on_cpu


@pytest.mark.parametrize('free,reserved,allocated,reserve,requested,expected',[
    (40,0,0,12,32,28),
    (8,44,20,12,32,20),
    (8,56,20,12,32,32),
    (8,56,52,12,32,0),
])
def test_gpu_budget_includes_only_reusable_allocator_memory(free,reserved,allocated,reserve,requested,expected):
    gib=2**30
    with patch('torch.cuda.is_available',return_value=True), \
         patch('torch.cuda.mem_get_info',return_value=(free*gib,80*gib)), \
         patch('torch.cuda.memory_reserved',return_value=reserved*gib), \
         patch('torch.cuda.memory_allocated',return_value=allocated*gib):
        with saved_on_cpu(keep_gib=requested,reserve_gib=reserve) as stats:
            assert stats['gpu_keep_budget_bytes']==expected*gib


def test_multiple_accumulation_updates_preserve_gradients():
    torch.manual_seed(42)
    baseline=torch.nn.Linear(8,4)
    actual=torch.nn.Linear(8,4);actual.load_state_dict(baseline.state_dict())
    a=torch.optim.AdamW(actual.parameters(),lr=1e-3)
    b=torch.optim.AdamW(baseline.parameters(),lr=1e-3)
    for _ in range(3):
        a.zero_grad();b.zero_grad()
        for _ in range(2):
            x=torch.randn(3,8)
            expected=baseline(x).square().mean()/2
            expected.backward()
            with saved_on_cpu() as stats:
                loss=actual(x).square().mean()/2
                loss.backward()
            torch.testing.assert_close(loss,expected)
            assert stats['saved_bytes']>0
        for p,q in zip(actual.parameters(),baseline.parameters()):torch.testing.assert_close(p.grad,q.grad)
        a.step();b.step()
    for p,q in zip(actual.parameters(),baseline.parameters()):torch.testing.assert_close(p,q)


@pytest.mark.parametrize('key',['profile_optimizer_steps','profile_gradient_accumulation_steps'])
def test_profile_settings_require_positive_integers(tmp_path,key):
    import json
    from train.src.infrastructure.config import load_config,ROOT
    c=json.loads((ROOT/'train/configs/minicpmo_full_sft_plain_ce_dp8.json').read_text())
    path=tmp_path/'config.json'
    for value in [0,-1,True,1.5]:
        path.write_text(json.dumps(dict(c,**{key:value})))
        with pytest.raises(ValueError,match=key):load_config(path)
    path.write_text(json.dumps(dict(c,**{key:2})))
    assert load_config(path)[key]==2
