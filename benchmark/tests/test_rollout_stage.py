import hashlib
import json
from pathlib import Path

import pytest

from benchmark.data.files import append_jsonl, read_jsonl, write_json
from benchmark.runtime.rollout_stage import prepare, parse_parallel, completed_rows, adopt_first, summarize


def fixture_config(tmp_path):
    data = tmp_path/'data.json'
    data.write_text(json.dumps([dict(task='RTD', video=f'{i}.mp4') for i in range(12)]))
    prompt = tmp_path/'prompt.txt'; prompt.write_text('streaming prompt')
    return dict(output=str(tmp_path/'diagnostics'), selection_seed=20261110, sample_count=5,
        rollout_seeds=[42, 43], full_command=['/usr/bin/python3', '-m', 'benchmark', 'parallel', 'minicpmo',
        '--data', str(data), '--system-prompt', str(prompt), '--videos-dir', str(tmp_path),
        '--model-path', str(tmp_path/'model'), '--output-dir', str(tmp_path/'full'), '--gpus', '0', '1'])


def fake_run_config(args, prompt):
    cfg = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    import os
    cfg['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    return cfg


def create_rollouts(config, manifest, commands):
    from benchmark.data.agent_inputs import input_protocol
    for number, command in enumerate(commands):
        args, controls = parse_parallel(command)
        args.output_dir.mkdir(parents=True)
        rows = []
        for rank, gpu in enumerate(controls['gpus']):
            target = args.output_dir/f'worker_{rank}'; target.mkdir()
            worker, _ = parse_parallel(command)
            worker.output_dir = target; worker.device = 'cuda:0'
            cfg = fake_run_config(worker, '')
            cfg.update(cuda_visible_devices=gpu, agent_input_protocol=input_protocol(worker),
                       agent_call_schema=worker.agent_call_schema)
            write_json(target/'run_config.json', cfg)
            for index in manifest['sample_indices'][rank::len(controls['gpus'])]:
                row = dict(sample_index=index, status='complete', events=[],
                           system_prompt_sha256=hashlib.sha256(b'rendered').hexdigest())
                append_jsonl(target/'predictions.jsonl', row); rows.append(row)
                append_jsonl(target/'sample_system_prompts.jsonl', dict(sample_index=index,system_prompt='rendered'))
                (target/f'sample_{index:03d}.chunks.jsonl').write_text('{"silent":true}\n')
        for row in rows: append_jsonl(args.output_dir/'predictions.jsonl', row)
        write_json(args.output_dir/'metrics.json', dict(per_sample=[dict(sample_index=r['sample_index'], hits=0) for r in rows]))


def test_pinned_uniform_selection_original_indices_and_seed_zero(tmp_path):
    c = fixture_config(tmp_path)
    m, commands = prepare(c)
    assert len(set(m['sample_indices'])) == 5
    assert prepare(c) == (m, commands)
    for number, command in enumerate(commands):
        args, controls = parse_parallel(command)
        assert args.seed == 42 + number
        assert args.data == Path(c['full_command'][6])
        assert controls['gpus'] == ['0','1']
    changed = dict(c, selection_seed=99)
    with pytest.raises(ValueError, match='manifest changed'): prepare(changed)


@pytest.mark.parametrize('updates', [dict(sample_count=13), dict(rollout_seeds=[43,44]),
                                    dict(rollout_seeds=[42,42])])
def test_invalid_design_rejected(tmp_path, updates):
    with pytest.raises(ValueError): prepare(dict(fixture_config(tmp_path), **updates))


def test_complete_trajectories_required(tmp_path):
    append_jsonl(tmp_path/'predictions.jsonl', dict(sample_index=0,status='partial'))
    with pytest.raises(ValueError, match='partial'): completed_rows(tmp_path,[0])


def test_adopt_fixed_first_seed_keeps_raw_prompts_and_is_resumable(tmp_path, monkeypatch):
    from benchmark.adapters import minicpmo
    monkeypatch.setattr(minicpmo,'run_config',fake_run_config)
    monkeypatch.setattr(minicpmo,'prepare_args',lambda a: None)
    c = fixture_config(tmp_path); m, commands = prepare(c)
    create_rollouts(c,m,commands)
    adopt_first(c,m); adopt_first(c,m)
    args, _ = parse_parallel(c['full_command'])
    rows = [r for d in args.output_dir.glob('worker_*') for r in read_jsonl(d/'predictions.jsonl')]
    assert len(rows) == 5
    for d in args.output_dir.glob('worker_*'):
        for r in read_jsonl(d/'predictions.jsonl'):
            assert (d/f"sample_{r['sample_index']:03d}.chunks.jsonl").exists()
    summarize(c,m)
    aggregate=read_jsonl(Path(c['output'])/'rollouts.jsonl')
    assert len(aggregate)==10
    assert all(r['sample_seed']==r['seed']+r['sample_index'] for r in aggregate)
    first=Path(c['output'])/'rollout_00/worker_0/run_config.json'
    cfg=json.loads(first.read_text());cfg['seed']=999;write_json(first,cfg)
    with pytest.raises(ValueError, match='configuration differs'): adopt_first(c,m)
