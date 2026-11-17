import json
import pytest
from train.scripts.sft_then_split_bench import validate, previous_ready


def test_existing_training_wait_uses_process_start_identity(monkeypatch):
    from train.scripts.sft_then_split_bench import training_processes_alive
    config = {'training_processes': [{'pid': 123, 'start': 'original'}]}
    monkeypatch.setattr('benchmark.scripts.watch_benchmark_queue.process',
                        lambda pid: {'start': 'original'})
    assert training_processes_alive(config)
    monkeypatch.setattr('benchmark.scripts.watch_benchmark_queue.process',
                        lambda pid: {'start': 'reused'})
    assert not training_processes_alive(config)
    monkeypatch.setattr('benchmark.scripts.watch_benchmark_queue.process', lambda pid: None)
    assert not training_processes_alive(config)


def test_benchmarks_only_never_launches_training_or_uses_partial_checkpoint(tmp_path, monkeypatch):
    from train.scripts import sft_then_split_bench as controller
    output = tmp_path/'existing-training'
    output.mkdir()
    (output/'run_config.json').write_text('{}')
    aoi = tmp_path/'aoi.json'
    aoi.write_text('{}')
    config = tmp_path/'chain.json'
    config.write_text(json.dumps(dict(training_output=str(output), training_processes=[],
        runtime_root=str(tmp_path), aoi_config=str(aoi), poll_seconds=1)))
    monkeypatch.setattr('sys.argv', ['controller', '--config', str(config), '--benchmarks-only'])
    monkeypatch.setattr(controller, 'validate', lambda *a, **kw: {})
    def incomplete(*args):
        raise ValueError('Final checkpoint is incomplete')
    monkeypatch.setattr(controller, 'completed_training', incomplete)
    def forbidden(*args, **kwargs):
        raise AssertionError('No training/export/inference may launch before final validation')
    monkeypatch.setattr(controller.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='incomplete'):
        controller.main()
    assert json.loads((tmp_path/'status.json').read_text())['stage'] == 'failed'


def test_eight_gpu_batch_and_disjoint_benchmark_pools(tmp_path):
    training = json.loads(open('train/configs/minicpmo_full_sft_plain_ce_dp.json').read())
    training['gradient_accumulation_steps'] = 2
    path = tmp_path / 'training.json'
    path.write_text(json.dumps(training))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        counts=dict(train=3996, validation=1004), videos=dict(train=['a'], validation=['b']))))
    (tmp_path / 'train.json').write_text(json.dumps([{}] * 3996))
    config = dict(training_config=str(path), prepared=str(tmp_path), expected_steps=500,
        train_gpus=list(map(str, range(8))), aoi_gpus=['0', '1', '2', '3'], omnipro_gpus=['4', '5', '6', '7'],
        training_output='/tmp/test-external-plain-ce', export_model='/tmp/test-external-plain-export')
    assert validate(config)['global_batch_size'] == 16
    restricted = dict(config, benchmark_gpus=['0','1','2','3'],
                      aoi_gpus=['0','1'], omnipro_gpus=['2','3'])
    # Evaluation allocation must not change the completed eight-GPU SFT batch.
    assert validate(restricted)['global_batch_size'] == 16
    with pytest.raises(ValueError, match='partition'):
        validate(dict(restricted, omnipro_gpus=['4','5']))
    config['omnipro_gpus'] = ['3', '4', '5', '6']
    with pytest.raises(ValueError, match='partition'):
        validate(config)
    del config['export_model']
    config['train_gpus']=['0','1','2','3']
    training['gradient_accumulation_steps']=4
    path.write_text(json.dumps(training))
    assert validate(config,train_only=True)['global_batch_size']==16
    config['continuation']=['python','--version']
    with pytest.raises(ValueError,match='Continuation'):
        validate(config,train_only=True)
    config['continuation_cwd']=str(tmp_path)
    assert validate(config,train_only=True)['global_batch_size']==16


def test_previous_judge_gate_accepts_terminal_item_failures_but_not_service_crash(tmp_path, monkeypatch):
    c = {'processes': []}
    states = dict(chain_status={'stage':'benchmarks_running'}, omnipro_status={'stage':'inference_running'},
                  aoi_status={'status':'benchmark_running'}, judge_stage={})
    for key, value in states.items():
        path=tmp_path/(key+'.json');path.write_text(json.dumps(value));c[key]=str(path)
    assert not previous_ready(c)
    for key,value in dict(chain_status={'stage':'complete'}, omnipro_status={'stage':'complete'},
                          aoi_status={'status':'complete'}, judge_stage={'status':'incomplete'}).items():
        open(c[key],'w').write(json.dumps(value))
    assert previous_ready(c)
    c['processes'] = [dict(pid=123, start='original')]
    monkeypatch.setattr('benchmark.scripts.watch_benchmark_queue.process', lambda pid: dict(start='original'))
    assert not previous_ready(c)
    monkeypatch.setattr('benchmark.scripts.watch_benchmark_queue.process', lambda pid: dict(start='reused'))
    assert previous_ready(c)
    open(c['judge_stage'],'w').write(json.dumps({'status':'running'}))
    with pytest.raises(RuntimeError,match='Judge'):
        previous_ready(c)
    open(c['chain_status'],'w').write(json.dumps({'stage':'failed'}))
    with pytest.raises(RuntimeError,match='failed/stopped'):
        previous_ready(c)


def test_resume_requires_complete_shards_rng_and_identical_config(tmp_path):
    from train.src.infrastructure.config import load_config
    training = json.loads(open('train/configs/minicpmo_full_sft_plain_ce_dp8.json').read())
    path=tmp_path/'sft.json';path.write_text(json.dumps(training))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(counts=dict(train=16,validation=1),videos=dict(train=['a'],validation=['b']))))
    (tmp_path/'train.json').write_text(json.dumps([{}]*16))
    checkpoint=tmp_path/'checkpoint-350';(checkpoint/'state').mkdir(parents=True)
    c=dict(training_config=str(path),prepared=str(tmp_path),train_gpus=list(map(str,range(8))),
           expected_steps=2,training_output=str(tmp_path/'new-output'),resume_checkpoint=str(checkpoint))
    with pytest.raises(ValueError,match='incomplete'):validate(c,train_only=True)
    for rank in range(8):
        (checkpoint/f'rng_rank{rank}.pt').write_bytes(b'fixture')
        (checkpoint/'state'/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt').write_bytes(b'fixture')
    (checkpoint/'state/mp_rank_00_model_states.pt').write_bytes(b'fixture')
    resolved=load_config(path)
    (checkpoint/'training_config.json').write_text(json.dumps(resolved))
    assert validate(c,train_only=True)['expected_steps']==2
    resolved['learning_rate']*=2
    (checkpoint/'training_config.json').write_text(json.dumps(resolved))
    with pytest.raises(ValueError,match='configuration differs'):validate(c,train_only=True)


@pytest.mark.parametrize('rollout_failure', [False, True])
def test_rollout_barrier_precedes_both_full_benchmarks(tmp_path, monkeypatch, rollout_failure):
    import hashlib
    from train.scripts import sft_then_split_bench as controller
    training = tmp_path/'training';training.mkdir();(training/'run_config.json').write_text('{}')
    output=tmp_path/'aoi';output.mkdir()
    model=tmp_path/'model';model.mkdir()
    metadata=model/'full_training_model.json'
    metadata.write_text(json.dumps(dict(source_checkpoint=str(training/'checkpoint-final'))))
    (output/'export_complete.json').write_text(json.dumps(dict(model=str(model),
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())))
    aoi=tmp_path/'aoi.json';aoi.write_text(json.dumps(dict(output=str(output),benchmark_python='python')))
    path=tmp_path/'chain.json';path.write_text(json.dumps(dict(training_output=str(training),
        training_processes=[],runtime_root=str(tmp_path),aoi_config=str(aoi),python='python',
        poll_seconds=0,rollout_configs=['aoi-rollouts.json','omni-rollouts.json'],
        export_model=str(model),live_reports='live.json')))
    monkeypatch.setattr('sys.argv',['controller','--config',str(path),'--benchmarks-only'])
    monkeypatch.setattr(controller,'validate',lambda *a,**kw: {})
    monkeypatch.setattr(controller,'completed_training',lambda *a: training/'checkpoint-final')
    monkeypatch.setattr('benchmark.runtime.parallel.terminate_owned',lambda *a: None)
    monkeypatch.setattr(controller.subprocess,'run',lambda *a,**kw: None)
    launched=[]
    class Child:
        def __init__(self,argv):self.argv=argv;self.pid=123;self.returncode=0
        def poll(self):return self.returncode
        def wait(self):return self.returncode
    def launch(argv,**kw):
        launched.append(argv)
        child=Child(argv)
        if 'aoi-rollouts.json' in argv and rollout_failure:child.returncode=1
        return child
    monkeypatch.setattr(controller.subprocess,'Popen',launch)
    if rollout_failure:
        with pytest.raises(RuntimeError,match='rollouts failed'):controller.main()
        assert len(launched)==3  # Export plus two diagnostics; no full inference.
    else:
        controller.main()
        assert '--export-only' in launched[0]
        assert launched[1][-1]=='aoi-rollouts.json'
        assert launched[2][-1]=='omni-rollouts.json'
        assert 'benchmark_after_training.py' in launched[3][1]
        assert '--export-only' not in launched[3]
        assert '--omnipro-only' in launched[4]
