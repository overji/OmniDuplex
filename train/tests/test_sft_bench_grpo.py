"""CPU checks for the post-SFT scheduler; no process signals or GPU allocation."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest

spec = importlib.util.spec_from_file_location('chain', Path(__file__).parents[1]/'scripts/sft_bench_grpo.py')
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def test_profile_requires_actual_finite_update(tmp_path):
    for rank in range(4):
        (tmp_path/f'memory_report.rank{rank}.json').write_text(json.dumps(dict(optimizer_steps=1,last_loss=.3)))
    chain.verify_profile(tmp_path,4)
    (tmp_path/'memory_report.rank3.json').write_text(json.dumps(dict(optimizer_steps=0,last_loss=.3)))
    with pytest.raises(RuntimeError):
        chain.verify_profile(tmp_path,4)


def test_split_does_not_leak_validation(tmp_path):
    a,b=tmp_path/'sft',tmp_path/'grpo'
    a.mkdir(); b.mkdir()
    manifest=dict(source_sha256='same',videos=dict(train=['a'],validation=['b']),counts=dict(train=1,validation=1))
    for p in (a,b):
        (p/'manifest.json').write_text(json.dumps(manifest))
    chain.verify_split(b,a)
    manifest['videos']['train']=['b']
    (b/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        chain.verify_split(b,a)


def test_reservation_only_claims_judge_and_restores(tmp_path):
    reservation=tmp_path/'reservation.json'
    judge=tmp_path/'judge.json';judge.write_text(json.dumps({'gpu':7}))
    journal=tmp_path/'lease.json'
    config=dict(gpu_reservations=str(reservation),judge_config=str(judge))
    with patch.object(chain.os,'kill') as kill:
        chain.acquire_lease(config,journal)
        assert chain.read(reservation)['gpus']==[7]
        chain.restore_lease(journal)
        assert chain.read(reservation)['gpus']==[]
        chain.restore_lease(journal)
        kill.assert_not_called()


def test_reservation_preserves_another_owner(tmp_path):
    reservation=tmp_path/'reservation.json'
    reservation.write_text(json.dumps(dict(owner='other',gpus=[7])))
    with pytest.raises(RuntimeError):
        chain.acquire_lease(dict(gpu_reservations=str(reservation)),tmp_path/'lease.json')
    assert chain.read(reservation)['owner']=='other'
