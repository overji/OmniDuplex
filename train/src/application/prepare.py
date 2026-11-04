"""按视频划分和按模式采样，写入prepared及可审计manifest；只组织数据，不加载模型。"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.domain.agent import PROTOCOL, training_sample
from train.src.domain.records import qa_record


def qa_identity(sample):
    """Stable content identity even when a larger source JSON reorders QA."""
    sample = training_sample(sample)
    fields = ('video', 'task', 'question', 'question_time', 'responses', 'available_tools')
    return hashlib.sha256(json.dumps({k: sample.get(k) for k in fields},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def video_split(videos, seed, fraction, split_from=None):
    """Preserve historical membership; hash only newly added video paths."""
    if split_from is None:
        shuffled = sorted(videos)
        random.Random(seed).shuffle(shuffled)
        return set(shuffled[:max(1, round(len(shuffled) * fraction))]), None
    previous = json.loads(split_from.read_text())
    train, val = (set(previous['videos'][key]) for key in ('train', 'validation'))
    if train & val:
        raise ValueError('Previous train/validation videos overlap')
    added = set(videos) - train - val
    val |= {video for video in added if int(hashlib.sha256(
        f'{seed}:{video}'.encode()).hexdigest(), 16) / 2**256 < fraction}
    return val & set(videos), previous


def prepare(data, output, config, max_samples, validation_fraction, *, split_from=None, audit_report=None):
    """按视频隔离后选择QA并写prepared/manifest；max_samples限制QA数，不把多个response拆为独立训练样本。"""
    if not 0 < validation_fraction < 1 or max_samples < 2:
        raise ValueError('Need >=2 samples and validation fraction between 0 and 1')
    if output.exists():
        raise ValueError('Use a new prepared-data directory')
    samples = json.loads(data.read_text())
    # All QA on the same UUID video stay together. Source aliases must be deduplicated upstream.
    videos = sorted({s['video'] for s in samples})
    validation, previous = video_split(videos, config['seed'], validation_fraction, split_from)
    previous_qa = set()
    if previous is not None:
        previous_qa = {qa_identity(r['sample']) for r in json.loads(
            (split_from.parent / 'validation.json').read_text())}
    audited_exclusions = {}
    if audit_report is not None:
        audit = json.loads(audit_report.read_text())
        if audit['source_sha256'] != hashlib.sha256(data.read_bytes()).hexdigest() or audit['config'] != config:
            raise ValueError('Audit source/config differs from preparation')
        accepted = [r['sample_index'] for r in audit['accepted']]
        excluded = [r['sample_index'] for r in audit['excluded']]
        if len(accepted + excluded) != len(samples) or set(accepted + excluded) != set(range(len(samples))):
            raise ValueError('Audit must cover every source QA exactly once')
        audited_exclusions = {r['sample_index']: r for r in audit['excluded']}
    buckets = {'train': defaultdict(list), 'validation': defaultdict(list)}
    skipped = Counter()
    exclusions = []
    for i, sample in enumerate(samples):
        if i in audited_exclusions:
            item = audited_exclusions[i]
            skipped[item['reason'].partition(':')[0]] += 1
            exclusions.append(dict(item, split='validation' if sample['video'] in validation else 'train'))
            continue
        record, reason = qa_record(sample, i, config)
        if audit_report is not None and reason:
            raise ValueError(f'Audit-accepted QA {i} is no longer eligible: {reason}')
        if record and config.get('training_method') == 'full_grpo':
            refs = reference_windows(sample)
            if config.get('grpo_horizon') != 'full_video' and any(seconds(r['window_end']) > config['max_prefix_seconds'] for _, r in refs):
                record, reason = None, 'window_exceeds_rollout_horizon'
            elif any(seconds(r['window_end']) < math.ceil(seconds(sample['question_time'])) + 1 for _, r in refs):
                record, reason = None, 'noncausal_window'
        if reason:
            skipped[reason.partition(':')[0]] += 1
            exclusions.append({'sample_index': i, 'reason': reason})
            continue
        split = 'validation' if sample['video'] in validation else 'train'
        buckets[split][sample['mode']].append(record)
    selected = {}
    candidate_counts = {split: sum(len(g) for g in groups.values()) for split, groups in buckets.items()}
    for split, groups in buckets.items():
        chosen = []
        rng = random.Random(config['seed'])
        for group in groups.values():
            rng.shuffle(group)
        limit = max_samples if split == 'train' else max(1, round(max_samples * validation_fraction))
        while len(chosen) < limit and any(groups.values()):
            for mode in sorted(groups):
                if groups[mode] and len(chosen) < limit:
                    chosen.append(groups[mode].pop())
        selected[split] = chosen
    if not all(selected.values()):
        raise ValueError('No eligible train/validation records; increase prefix limit or use more videos')
    output.mkdir(parents=True)
    for split, records in selected.items():
        (output / f'{split}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    from train.src.domain.agent import RECORD_FORMAT
    manifest = {'protocol': PROTOCOL, 'record_format': RECORD_FORMAT,
        'source': str(data.resolve()), 'source_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'config': config, 'counts': {s: len(r) for s, r in selected.items()}, 'skipped': dict(skipped),
        'response_counts': {s: sum(len(r['targets']) for r in rows) for s, rows in selected.items()},
        'source_scored_responses': sum(len(reference_windows(s)) for s in samples),
        'exclusions': exclusions,
        'candidate_counts': candidate_counts,
        'sampling_omissions': {s: candidate_counts[s] - len(rows) for s, rows in selected.items()},
        'videos': {s: sorted({r['sample']['video'] for r in rows}) for s, rows in selected.items()},
        'scope': ('All scored windows within full rollout horizon. No tool results.'
                  if config.get('training_method') == 'full_grpo' else
                  'All scored responses per QA in one continuous trajectory; full prefix from time zero. No tool results.')}
    if split_from is not None:
        manifest.update(split_from=str(split_from.resolve()), split_from_sha256=hashlib.sha256(split_from.read_bytes()).hexdigest(),
                        split_policy='preserve_existing_hash_new_videos_v1', validation_groups={
                            'previous': [r['sample_index'] for r in selected['validation'] if qa_identity(r['sample']) in previous_qa],
                            'added': [r['sample_index'] for r in selected['validation'] if qa_identity(r['sample']) not in previous_qa]})
    if audit_report is not None:
        manifest.update(audit_report=str(audit_report.resolve()), audit_sha256=hashlib.sha256(audit_report.read_bytes()).hexdigest())
        if not any(manifest['sampling_omissions'].values()):
            assert sum(manifest['response_counts'].values()) + sum(r['response_count'] for r in audit['excluded']) == audit['source_responses']
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('counts', 'skipped', 'scope')}, indent=2))
