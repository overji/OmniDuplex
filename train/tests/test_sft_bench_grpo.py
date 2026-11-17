"""CPU checks for the post-SFT scheduler; no process signals or GPU allocation."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest

spec = importlib.util.spec_from_file_location('chain', Path(__file__).parents[1]/'scripts/sft_bench_grpo.py')
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def test_no_judge_backend_needs_no_service_configuration(tmp_path):
    path=tmp_path/'grpo.json';path.write_text(json.dumps({'reward_backend':'tp_full_credit'}))
    config=dict(grpo_config=str(path),profile_config=str(path))
    assert chain.needs_judge_service(config) is False
    other=tmp_path/'profile.json';other.write_text(json.dumps({'reward_backend':'llm'}))
    with pytest.raises(ValueError,match='backends differ'):
        chain.needs_judge_service(dict(config,profile_config=str(other)))
    assert chain.needs_judge_service(dict(grpo_config=str(other),profile_config=str(other))) is True


@pytest.mark.parametrize('check_only',[True,False])
def test_no_judge_controller_skips_service_gpu_checks_and_leases(tmp_path,monkeypatch,check_only):
    from types import SimpleNamespace
    scoring=tmp_path/'scoring.json';scoring.write_text(json.dumps({'reward_backend':'tp_full_credit'}))
    env=tmp_path/'empty.env';env.write_text('')
    config=dict(output=str(tmp_path/'controller'),grpo_config=str(scoring),profile_config=str(scoring),
        profile_output=str(tmp_path/'profile'),grpo_output=str(tmp_path/'grpo'),
        prepared=str(tmp_path/'prepared'),sft_prepared=str(tmp_path/'prepared'),
        train_gpus=[4,5,6,7],python='/usr/bin/python',env_file=str(env),poll_seconds=1)
    path=tmp_path/'chain.json';path.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv',['chain','--config',str(path),'--grpo-only']+(['--check-only'] if check_only else []))
    commands=[]
    def popen(argv,**kwargs):
        commands.append(argv)
        return SimpleNamespace(pid=123456,returncode=0,poll=lambda:0)
    with patch.object(chain,'verify_split',return_value={'train':2,'validation':1}), \
         patch.object(chain,'verified_sft_export',return_value=tmp_path/'model'), \
         patch.object(chain,'verify_profile'), \
         patch.object(chain,'acquire_lease',side_effect=AssertionError('No Judge lease')), \
         patch.object(chain.subprocess,'check_output',side_effect=AssertionError('No Judge GPU query')), \
         patch.object(chain.subprocess,'Popen',side_effect=popen):
        chain.main()
    assert len(commands)==(0 if check_only else 2)
    assert not any('serve_grpo_judge.py' in ' '.join(cmd) for cmd in commands)
    assert not (tmp_path/'controller/gpu_lease.json').exists()


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


def test_grpo_only_requires_verified_export_and_matching_training_model(tmp_path):
    import hashlib
    model=tmp_path/'model';model.mkdir();checkpoint=tmp_path/'training/checkpoint-final'
    metadata=model/'full_training_model.json';metadata.write_text(json.dumps(dict(source_checkpoint=str(checkpoint))))
    out=tmp_path/'bench';out.mkdir()
    marker=out/'export_complete.json';marker.write_text(json.dumps(dict(model=str(model),metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())))
    b=tmp_path/'benchmark.json';b.write_text(json.dumps(dict(export_model=str(model),output=str(out))))
    g=tmp_path/'grpo.json';g.write_text(json.dumps(dict(model_path=str(model))))
    config=dict(benchmark_config=str(b),grpo_config=str(g),profile_config=str(g))
    with patch('train.scripts.benchmark_after_training.completed_training',return_value=checkpoint):
        assert chain.verified_sft_export(config)==model
        g.write_text(json.dumps(dict(model_path=str(tmp_path/'wrong'))))
        with pytest.raises(ValueError,match='model differs'):chain.verified_sft_export(config)
        marker.write_text('{}')
        with pytest.raises(ValueError,match='not verified'):chain.verified_sft_export(config)


def test_multi_update_profile_requires_exact_number_on_every_rank(tmp_path):
    for rank in range(2):
        (tmp_path/f'memory_report.rank{rank}.json').write_text(json.dumps(dict(optimizer_steps=2,last_loss=.8)))
    chain.verify_profile(tmp_path,2,optimizer_steps=2)
    (tmp_path/'memory_report.rank1.json').write_text(json.dumps(dict(optimizer_steps=1,last_loss=.8)))
    with pytest.raises(RuntimeError,match='2 finite updates'):
        chain.verify_profile(tmp_path,2,optimizer_steps=2)
