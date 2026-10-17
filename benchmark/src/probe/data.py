"""Construct paired probes without exposing reference annotations to models."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from benchmark.config.defaults import DEFAULTS
from benchmark.data.files import atomic_write_text, read_jsonl, write_json
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import is_scored_response, reference_windows
from benchmark.probe.media import scan_video, source_identity


SCHEMA = 'omniduplex-probe-v1'
REQUEST_FIELDS = ('id', 'task', 'mode', 'ability', 'question', 'question_time',
                  'available_tools', 'frontend_context')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_time(value):
    result = seconds(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid reference time')
    return result


def sample_probes(sample, sample_index, data_sha256, video, media):
    """Return rows plus explicit exclusions; ordinal counts include excluded pairs."""
    rows, excluded = [], []
    kind = sample.get('sample_type', (sample.get('generation_context') or {}).get('sample_type'))
    if sample.get('mode') == 'agent' and kind in {'no_context', 'missing_tool', 'negative'}:
        return [], [{'sample_index': sample_index, 'reason': 'legacy_agent_negative'}]
    qtime = finite_time(sample['question_time'])
    if not isinstance(sample.get('question'), str) or not sample['question'].strip():
        raise ValueError('Missing question')
    windows = sorted((finite_time(response['window_start']), finite_time(response['window_end']), index)
                     for index, response in reference_windows(sample))
    window_indices = {index for _, _, index in windows}
    for index, response in enumerate(sample['responses']):
        if is_scored_response(sample, response) and index not in window_indices:
            excluded.append({'sample_index': sample_index, 'response_index': index,
                             'reason': 'missing_reference_window'})
    if any(not qtime <= start <= end <= media['duration'] for start, end, _ in windows):
        raise ValueError('A reference window is outside the question/video bounds; prior count is unreliable')
    request = {key: sample[key] for key in REQUEST_FIELDS if key in sample}
    for ordinal, (start, end, index) in enumerate(windows):
        pre = start - DEFAULTS['probe']['pre_seconds']
        reason = None
        if pre <= qtime or pre <= 0:
            reason = 'negative_prefix_not_after_question'
        elif any(previous[0] >= pre for previous in windows[:ordinal]):
            reason = 'prior_opportunity_not_observed_before_negative_cutoff'
        if reason:
            excluded.append({'sample_index': sample_index, 'response_index': index, 'reason': reason})
            continue
        pair_id = digest([SCHEMA, data_sha256, sample_index, index, pre, end])
        for side, cutoff, expected in (('pre', pre, 'NO'), ('post', end, 'YES')):
            rows.append({'schema': SCHEMA, 'probe_id': pair_id + ':' + side, 'pair_id': pair_id,
                'sample_index': sample_index, 'response_index': index, 'side': side,
                'expected': expected, 'prefix_end_seconds': cutoff, 'handled_count': ordinal,
                'window_start_seconds': start, 'window_end_seconds': end,
                'task': sample['task'], 'mode': sample['mode'], 'ability': sample['ability'],
                'video': str(video.resolve()), 'source': media['source'],
                'has_audio': media['has_audio'], 'duration_seconds': media['duration'],
                'request': request})
    return rows, excluded


def prepare(args):
    data_sha256 = file_hash(args.data)
    samples = json.loads(args.data.read_text())
    if not isinstance(samples, list):
        raise ValueError('QA data must be a JSON array')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use an empty probe manifest output directory')
    rows, excluded, sources = [], [], {}
    for index, sample in enumerate(samples):
        try:
            video = (args.videos_dir / sample['video']).resolve()
            if video not in sources:
                sources[video] = scan_video(video)
                sources[video]['source'] = source_identity(video)
            probes, rejected = sample_probes(sample, index, data_sha256, video, sources[video])
            rows.extend(probes)
            excluded.extend(rejected)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            excluded.append({'sample_index': index, 'reason': 'invalid_sample_or_missing_media', 'error': str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'probe_manifest.jsonl'
    atomic_write_text(manifest, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    atomic_write_text(args.output_dir / 'probe_exclusions.jsonl',
                      ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in excluded))
    summary = {'schema': SCHEMA, 'source_samples': len(samples), 'eligible_pairs': len(rows) // 2,
               'probe_count': len(rows), 'excluded_records': len(excluded),
               'exclusion_reasons': dict(Counter(row['reason'] for row in excluded))}
    write_json(args.output_dir / 'probe_manifest_config.json', {
        'schema': SCHEMA, 'data': str(args.data.resolve()), 'data_sha256': data_sha256,
        'videos_dir': str(args.videos_dir.resolve()), 'pre_seconds': DEFAULTS['probe']['pre_seconds'],
        'positive_cutoff': 'window_end', 'prefix_interval': '[0, cutoff)',
        'manifest_sha256': file_hash(manifest)})
    write_json(args.output_dir / 'probe_prepare_report.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


def load_manifest(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Probe manifest has no eligible pairs')
    ids, pairs = set(), {}
    for row in rows:
        if row.get('schema') != SCHEMA or row.get('side') not in {'pre', 'post'}:
            raise ValueError('Unsupported probe manifest row')
        if row['probe_id'] in ids or row['expected'] != ('NO' if row['side'] == 'pre' else 'YES'):
            raise ValueError('Duplicate or mislabeled probe row')
        ids.add(row['probe_id'])
        pair = pairs.setdefault(row['pair_id'], {})
        if row['side'] in pair:
            raise ValueError('Duplicate probe side in pair')
        pair[row['side']] = row
    for pair in pairs.values():
        if set(pair) != {'pre', 'post'}:
            raise ValueError('Every manifest pair must have pre and post probes')
        pre, post = pair['pre'], pair['post']
        shared = ('sample_index', 'response_index', 'handled_count', 'request', 'video', 'source',
                  'window_start_seconds', 'window_end_seconds', 'duration_seconds', 'has_audio',
                  'task', 'mode', 'ability')
        if any(pre[key] != post[key] for key in shared):
            raise ValueError('Inconsistent pair metadata')
        if (pre['prefix_end_seconds'] != pre['window_start_seconds'] - DEFAULTS['probe']['pre_seconds']
                or post['prefix_end_seconds'] != post['window_end_seconds']
                or not finite_time(pre['request']['question_time']) < pre['prefix_end_seconds']
                < pre['window_start_seconds'] <= post['prefix_end_seconds'] <= pre['duration_seconds']):
            raise ValueError('Invalid probe cutoff bounds')
    return rows
