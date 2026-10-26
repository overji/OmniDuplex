"""Causal, multi-event labels. Source annotations are never rewritten."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import reference_windows
from train.src.agent import training_sample
from .config import PROTOCOL, write


def target(reference):
    from train.src.agent import target_text
    return target_text(reference)


def record(sample, index, horizon):
    sample = training_sample(sample)
    q = seconds(sample['question_time'])
    windows = reference_windows(sample)
    if not windows:
        return None, 'no_scored_windows'
    events = []
    for ri, ref in windows:
        start, end = seconds(ref['window_start']), seconds(ref['window_end'])
        evidence = max([seconds(e.get('available_time', e.get('time', 0))) for e in ref.get('evidence', [])] or [0])
        confirmed = seconds((ref.get('trigger') or {}).get('confirmed_time', 0))
        onset = math.ceil(max(start, math.floor(q) + 2 if q != math.floor(q) else q + 1,
                              evidence, confirmed, seconds(ref.get('trigger_time', start))))
        if onset > end:
            return None, 'noncausal_event'
        if end > horizon:
            return None, 'event_exceeds_horizon'
        events.append(dict(second=onset, response_index=ri, target=target(ref)))
    if len({e['second'] for e in events}) != len(events):
        return None, 'colliding_events'
    labels = []
    for t in range(1, horizon + 1):
        if any(e['second'] == t for e in events):
            labels.append(1)
        elif t - 1 < q or any(seconds(r['window_start']) <= t <= seconds(r['window_end']) for _, r in windows):
            labels.append(-100)
        else:
            labels.append(0)
    return dict(sample_index=index, sample=sample, horizon=horizon, events=events, labels=labels), None


def sampled_labels(row, config):
    labels = row['labels'][:]
    positive = [t for t, value in enumerate(labels, 1) if value == 1]
    negatives = [t for t, value in enumerate(labels, 1) if value == 0]
    limit = min(len(negatives), config['negative_ratio'] * len(positive))
    rng = random.Random(config['seed'] + row['sample_index'])
    nearby = [t for t in negatives if any(abs(t - p) <= config['boundary_seconds'] for p in positive)]
    rng.shuffle(nearby)
    keep = set(nearby[:limit // 2])
    others = [t for t in negatives if t not in keep]
    rng.shuffle(others); keep.update(others[:limit - len(keep)])
    return [value if value != 0 or t in keep else -100 for t, value in enumerate(labels, 1)]


def prepare(data, output, config):
    if output.exists():
        raise ValueError('Use a fresh prepared directory')
    samples = json.loads(data.read_text())
    videos = sorted({s['video'] for s in samples})
    random.Random(config['seed']).shuffle(videos)
    nv, nt = max(1, round(len(videos)*config['validation_fraction'])), max(1, round(len(videos)*config['test_fraction']))
    if nv + nt >= len(videos):
        raise ValueError('Need enough distinct videos for three splits')
    split = {v: ('validation' if i < nv else 'test' if i < nv + nt else 'train') for i, v in enumerate(videos)}
    rows = {s: [] for s in ('train', 'validation', 'test')}; excluded = []
    from benchmark.data.media import media_cache
    for i, sample in enumerate(samples):
        video = Path(config['videos_dir']) / sample['video']
        if not video.is_file():
            excluded.append(dict(sample_index=i, reason='missing_video')); continue
        try:
            _, metadata, _ = media_cache(video, Path(config['cache_dir']) / 'media')
            horizon = min(config['max_prefix_seconds'], math.floor(metadata['duration']))
            row, reason = record(sample, i, horizon)
        except (ValueError, KeyError) as exc:
            row, reason = None, f'invalid_annotation:{type(exc).__name__}'
        if row:
            row['train_labels'] = sampled_labels(row, config)
            rows[split[sample['video']]].append(row)
        else:
            excluded.append(dict(sample_index=i, reason=reason))
    if not all(rows.values()):
        raise ValueError('No eligible data in one split; increase horizon or provide more videos')
    for name, items in rows.items():
        write(output / f'{name}.json', items)
    write(output / 'excluded.json', excluded)
    write(output / 'manifest.json', dict(protocol=PROTOCOL, config=config,
        source=str(data.resolve()), source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        counts={k:len(v) for k,v in rows.items()}, excluded=dict(Counter(e['reason'] for e in excluded)),
        videos={k:sorted({r['sample']['video'] for r in v}) for k,v in rows.items()}))


def load_rows(directory, split, config):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Prepared protocol mismatch; prepare again')
    for key in ('seed', 'videos_dir', 'max_prefix_seconds', 'negative_ratio', 'boundary_seconds', 'validation_fraction', 'test_fraction'):
        if manifest['config'][key] != config[key]:
            raise ValueError('Prepared configuration changed: ' + key)
    return json.loads((directory / f'{split}.json').read_text())
