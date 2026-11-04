import json
import pytest
from train.scripts.sft_then_split_bench import validate, previous_ready


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
    config['omnipro_gpus'] = ['3', '4', '5', '6']
    with pytest.raises(ValueError, match='partition'):
        validate(config)


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
