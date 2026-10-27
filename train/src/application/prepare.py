"""按视频划分和按模式采样，写入prepared及可审计manifest；只组织数据，不加载模型。"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.domain.agent import PROTOCOL
from train.src.domain.records import qa_record


def prepare(data, output, config, max_samples, validation_fraction):
    """按视频隔离后选择QA并写prepared/manifest；max_samples限制QA数，不把多个response拆为独立训练样本。"""
    if not 0 < validation_fraction < 1 or max_samples < 2:
        raise ValueError('Need >=2 samples and validation fraction between 0 and 1')
    if output.exists():
        raise ValueError('Use a new prepared-data directory')
    samples = json.loads(data.read_text())
    # All QA on the same UUID video stay together. Source aliases must be deduplicated upstream.
    videos = sorted({s['video'] for s in samples})
    random.Random(config['seed']).shuffle(videos)
    nval = max(1, round(len(videos) * validation_fraction))
    validation = set(videos[:nval])
    buckets = {'train': defaultdict(list), 'validation': defaultdict(list)}
    skipped = Counter()
    exclusions = []
    for i, sample in enumerate(samples):
        record, reason = qa_record(sample, i, config)
        if record and config.get('training_method') == 'full_grpo':
            refs = reference_windows(sample)
            if any(seconds(r['window_end']) > config['max_prefix_seconds'] for _, r in refs):
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
        'videos': {s: sorted({r['sample']['video'] for r in rows}) for s, rows in selected.items()},
        'scope': ('All scored windows within full rollout horizon. No tool results.'
                  if config.get('training_method') == 'full_grpo' else
                  'All scored responses per QA in one continuous trajectory; full prefix from time zero. No tool results.')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('counts', 'skipped', 'scope')}, indent=2))
