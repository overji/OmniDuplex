import hashlib
import json
from pathlib import Path

import pytest
from train.src.application.prepare import prepare, video_split
from train.src.domain.agent import PROTOCOL


def fixture(tmp_path):
    samples = []
    for i in range(12):
        (tmp_path / f'{i}.mp4').touch()
        samples.append(dict(video=f'{i}.mp4', mode='pr', task='grounding_pr', question='Notify me.',
            question_time='00:00', responses=[dict(response='Ready.', trigger_time='00:02',
                window_start='00:02', window_end='00:04')]))
    source = tmp_path / 'source.json'; source.write_text(json.dumps(samples))
    previous = tmp_path / 'previous'; previous.mkdir()
    split = previous / 'manifest.json'
    split.write_text(json.dumps(dict(videos=dict(train=['0.mp4'], validation=['1.mp4']))))
    (previous / 'validation.json').write_text(json.dumps([dict(sample=samples[1])]))
    config = dict(videos_dir=str(tmp_path), seed=42, max_prefix_seconds=300,
                  training_method='full_sft', agent_output_protocol=PROTOCOL, sft_horizon='full_video')
    return samples, source, split, config


def test_old_membership_survives_expansion_and_qa_reordering(tmp_path):
    samples, source, split, config = fixture(tmp_path)
    first, _ = video_split([s['video'] for s in samples], 42, .2, split)
    second, _ = video_split([s['video'] for s in reversed(samples)] + ['later.mp4'], 42, .2, split)
    assert first == second - {'later.mp4'}
    assert '1.mp4' in first and '0.mp4' not in first
    source.write_text(json.dumps(list(reversed(samples))))
    prepare(source, tmp_path / 'out', config, 100000, .2, split_from=split)
    m = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert sum(m['counts'].values()) == len(samples)
    assert not any(m['sampling_omissions'].values())
    assert m['validation_groups']['previous'] == [10]  # new source index, same historical QA
    assert not set(m['videos']['train']) & set(m['videos']['validation'])


def test_exact_audit_exclusions_and_response_conservation(tmp_path):
    samples, source, split, config = fixture(tmp_path)
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), config=config,
        source_responses=12, accepted=[dict(sample_index=i) for i in range(11)],
        excluded=[dict(sample_index=11, reason='target_exceeds_video_duration', response_count=1)])
    path = tmp_path / 'audit.json';path.write_text(json.dumps(report))
    prepare(source, tmp_path / 'out', config, 100000, .2, split_from=split, audit_report=path)
    m = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert sum(m['counts'].values()) == 11
    assert sum(m['response_counts'].values()) + m['exclusions'][0]['response_count'] == 12
    assert m['skipped'] == {'target_exceeds_video_duration': 1}
    report['accepted'].append(dict(sample_index=0));path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='exactly once'):
        prepare(source, tmp_path / 'bad', config, 100000, .2, audit_report=path)


def test_sampling_cap_is_visible_and_source_mismatch_is_rejected(tmp_path):
    samples, source, split, config = fixture(tmp_path)
    prepare(source, tmp_path / 'small', config, 2, .2, split_from=split)
    m = json.loads((tmp_path / 'small/manifest.json').read_text())
    assert sum(m['sampling_omissions'].values()) > 0
    report = tmp_path / 'bad.json';report.write_text(json.dumps(dict(source_sha256='wrong', config=config)))
    with pytest.raises(ValueError, match='source/config'):
        prepare(source, tmp_path / 'out', config, 100000, .2, audit_report=report)
