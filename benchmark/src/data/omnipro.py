"""Lossless OmniPro benchmark conversion; only AOI timing windows are derived."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
from benchmark.data.files import write_json
from benchmark.data.timestamps import seconds

PAPER = 'https://arxiv.org/html/2605.18577v1#S3.SS2.SSS1'
# These are explicitly derived AOI interaction labels, not OmniPro annotations.
TASKS = {
    'instant_event_alert': ('pr', 'perception'),
    'explicit_target_grounding': ('pr', 'perception'),
    'snapshot_counting': ('pr', 'perception'),
    'realtime_state_monitor': ('rtd', 'perception'),
    'semantic_condition_alert': ('pr', 'comprehension'),
    'cumulative_counting': ('rtd', 'comprehension'),
    'event_narration': ('rtd', 'comprehension'),
    'dedup_counting': ('rtd', 'reasoning'),
    'sequential_step_instruction': ('pr', 'reasoning'),
}


def convert(rows):
    """Preserve order/IDs and source fields; never merge overlapping windows."""
    result, seen = [], set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate OmniPro ID')
        seen.add(row['id'])
        mode, level = TASKS[row['task']]
        question = seconds(row['question_time'])
        if 'question_time_sec' in row and question != row['question_time_sec']:
            raise ValueError('Question timestamps disagree')
        responses = []
        for gt in row['ground_truth']:
            t = gt['trigger_time_sec']
            if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
                raise ValueError('Invalid numeric trigger timestamp')
            responses.append(dict(response=gt['response'], trigger_time=t,
                window_start=max(0., t-3), window_end=t+3,
                trigger_type=gt.get('trigger_type'), details=dict(gt), source_reference=dict(gt)))
        if not responses:
            raise ValueError('OmniPro QA has no reference triggers')
        # Top-level source fields remain intact; only standard AOI fields are added.
        sample = dict(row, video=row['video_path'], mode=mode, ability=row['task'],
                      cognitive_level=level, responses=responses, source_dataset='OmniPro',
                      interaction_type_is_derived=True, source_sample=dict(row))
        result.append(sample)
    return result


def video_duration(path):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True))


def prepare(source, videos_dir, output, training_metadata=None):
    source, videos_dir, output = Path(source), Path(videos_dir), Path(output)
    source_bytes = source.read_bytes()
    samples = convert(json.loads(source_bytes))
    videos = sorted({s['video'] for s in samples})
    missing = [v for v in videos if not (videos_dir/v).is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} OmniPro videos; no silent exclusions')
    with ThreadPoolExecutor(max_workers=4) as pool:
        durations = dict(zip(videos, pool.map(video_duration, [videos_dir/v for v in videos])))
    for sample in samples:
        sample['duration_seconds'] = durations[sample['video']]
    annotations = []
    for i, sample in enumerate(samples):
        for ri, gt in enumerate(sample['ground_truth']):
            try: agrees = seconds(gt['trigger_time']) == gt['trigger_time_sec']
            except (ValueError, TypeError): agrees = False
            if not agrees:
                annotations.append(dict(sample_index=i,response_index=ri,source_time=gt['trigger_time'],used_seconds=gt['trigger_time_sec']))
    anomalies = []
    for i, s in enumerate(samples):
        for ri, response in enumerate(s['responses']):
            t = response['trigger_time']; duration = durations[s['video']]
            if t > duration:
                anomalies.append(dict(sample_index=i, id=s['id'], response_index=ri,
                    trigger_time=t, actual_duration=duration,
                    window_unreachable=response['window_start'] > duration))
    overlap = dict(checked=False)
    if training_metadata:
        training_metadata = Path(training_metadata)
        meta = json.loads(training_metadata.read_text())
        originals = {Path(v['original_file']).stem for v in meta['videos'] if v.get('original_file')}
        shared = sorted({s['video_id'] for s in samples} & originals)
        overlap = dict(checked=True, method='original source video ID intersection; not perceptual deduplication',
            metadata=str(training_metadata), metadata_sha256=hashlib.sha256(training_metadata.read_bytes()).hexdigest(),
            shared_source_video_ids=shared, qa_indices=[i for i,s in enumerate(samples) if s['video_id'] in shared])
    manifest = dict(version='omnipro_aoi_v1', source=str(source.resolve()),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(), paper=PAPER, tolerance_seconds=3,
        videos_dir=str(videos_dir.resolve()), samples=len(samples), videos=len(videos),
        references=sum(len(s['responses']) for s in samples),
        tasks=dict(Counter(s['task'] for s in samples)), interaction_mapping={k:list(v) for k,v in TASKS.items()},
        anomalies=anomalies, timestamp_annotations=annotations, training_overlap=overlap,
        scoring='AOI timing-only inclusive one-to-one onset matching; not original content-gated OmniPro F1')
    output.mkdir(parents=True, exist_ok=True)
    payloads = {'all_qa.json': samples, 'manifest.json': manifest,
                'source_mapping.json': [dict(sample_index=i,id=s['id'],video=s['video']) for i,s in enumerate(samples)]}
    for name, value in payloads.items():
        path = output/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Converted inputs changed; choose a fresh output directory')
    for name, value in payloads.items():
        write_json(output/name, value)
    return manifest
