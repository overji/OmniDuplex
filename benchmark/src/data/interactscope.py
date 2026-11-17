"""Lossless Omni-InteractScope conversion into an explicitly derived AOI protocol."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from benchmark.data.files import atomic_write_text, write_json
from benchmark.data.timestamps import seconds

VERSION = 'interactscope_aoi_v1'
SOURCE = 'Omni-InteractScope'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_asset(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Asset escapes the dataset root')
    return path


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Missing or nonfinite annotation time')
    return value / 1000.


def tool_names(text):
    """Explicit name markers only; prose mentioning a name is not a call."""
    return sorted(set(re.findall(r'<call:([A-Za-z_][A-Za-z_0-9]*)>', text)))


def convert(row):
    """Keep all source fields separately. Crop offsets are NOT subtracted again."""
    task, sub = row['task'], row.get('sub_task_type', '')
    if task not in {f'T{i}' for i in range(1, 9)}:
        raise ValueError('Unknown task group')
    query_free = task == 'T8' and sub == 'A'
    query = 0. if query_free else finite(row['query_time_ms'])
    duration = row['video_duration_sec']
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 <= query < duration:
        raise ValueError('Query outside media')
    if not query_free and not row.get('query_text', '').strip():
        raise ValueError('Text input unavailable; never replace audio with annotation text')
    refs = []
    for source in row['responses']:
        trigger = finite(source['notify_trigger_time_ms'])
        if task == 'T8' and sub == 'A':
            window = source.get('valid_response_window_ms')
            if not isinstance(window, list) or len(window) != 2:
                raise ValueError('Missing annotated T8-A interval')
            start, end = map(finite, window)
        else:
            start, end = trigger - 2., trigger + 2.
            if task == 'T8':
                start = max(query, start)
        if start > end or trigger < 0:
            raise ValueError('Invalid window')
        text = source['assistant_reply_text']
        names = tool_names(text) if task == 'T1' else []
        if task == 'T1' and not names:
            raise ValueError('Tool target has no explicit reference tool names')
        refs.append(dict(response=text, window_start=start, window_end=end,
                         trigger_time=trigger, reference_tool_names=names, source_reference=source))
    if not refs:
        raise ValueError('No reference targets')
    return dict(id=row['interaction_id'], task=task + ('-' + sub if task == 'T8' else ''),
                mode='tool' if task == 'T1' else 'rtd' if task in ('T4', 'T7') or (task == 'T8' and sub == 'C') else 'pr',
                ability=task, video=row['video'], question=row.get('query_text', ''), question_time=query,
                query_free=query_free, split_query_boundary=True, duration_seconds=duration,
                responses=refs, available_tools=[], source_dataset=SOURCE, source_sample=row,
                frontend_context=dict(system_prompt=profile(row)))


def profile(row):
    """Only declared user needs, never trigger reasons, reference times or answers."""
    if row['task'] != 'T8':
        return ''
    group = row.get('target_group')
    if group not in ('visually_impaired', 'hearing_impaired') or row.get('qa_language') not in ('en', 'zh'):
        raise ValueError('Missing user profile or reply language')
    need = ('The user cannot see the scene. Convey relevant visual information in words.' if group == 'visually_impaired'
            else 'The user cannot hear the scene. Convey relevant audible events and speech in text.')
    policies = dict(A='Monitor for useful assistance and timely safety reminders without waiting for a question.',
                    B='Answer the user request when the observed evidence is sufficient.',
                    C='Guide the user with the currently actionable step as the observed state changes; do not give future steps.')
    return need + '\nReply in ' + ('English.' if row['qa_language'] == 'en' else 'Chinese.') + '\n' + policies[row['sub_task_type']]


def build_prompt(sample):
    """The runner passes only model-facing fields; source_sample is never needed."""
    from benchmark.config.paths import CONFIGS
    prompt = (CONFIGS / 'prompts/interactscope.txt').read_text().strip()
    context = (sample.get('frontend_context') or {}).get('system_prompt', '')
    if context:
        prompt += '\n\nUser context:\n' + context
    if sample.get('mode') == 'tool':
        definitions = (sample.get('frontend_context') or {}).get('tool_definitions', {})
        prompt += ('\n\nWhen a tool is warranted, use explicit <call:NAME> markers, replacing NAME with an available tool name. '
                   'You may give a brief spoken explanation after the markers. Never invent or translate tool names. '
                   'No tool will execute and no tool result will be supplied. Do not announce successful execution.\nAvailable tools:\n')
        prompt += '\n'.join(f'- {name}: {spec["desc"]}' for name, spec in definitions.items())
    if sample.get('query_free'):
        prompt += '\nThe standing user needs apply from stream start. Do not wait for an explicit question.'
    else:
        prompt += '\nBefore the user request arrives, remain silent. Do not acknowledge receipt of the request.'
    return prompt


def window_conflicts(sample):
    refs = sample['responses']
    return [dict(response_indices=[i, j], overlap_seconds=min(a['window_end'], b['window_end']) - max(a['window_start'], b['window_start']))
            for i, a in enumerate(refs) for j, b in enumerate(refs) if j > i
            and max(a['window_start'], b['window_start']) <= min(a['window_end'], b['window_end'])]


def unreachable_windows(sample):
    """Diagnostic only: source intervals may contain no post-query AOI tick."""
    first = math.floor(sample['question_time']) + 1
    duration = sample['duration_seconds']
    result = []
    for ri, ref in enumerate(sample['responses']):
        lo, hi = max(first, ref['window_start']), min(duration, ref['window_end'])
        integer_tick = lo <= hi and math.ceil(lo) <= math.floor(hi)
        final_tick = sample['question_time'] < duration and ref['window_start'] <= duration <= ref['window_end']
        if not integer_tick and not final_tick:
            result.append(ri)
    return result


def probe_video(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], text=True))
    if not any(s['codec_type'] == 'video' for s in data['streams']):
        raise ValueError('No video stream')
    return dict(duration=float(data['format']['duration']), has_audio=any(s['codec_type'] == 'audio' for s in data['streams']))


def prepare(config, output):
    """Verify available media, report every missing record and export fixed cohorts."""
    root, output = Path(config['dataset_root']).resolve(), Path(output).resolve()
    annotation = root / 'annotations.json'
    document = json.loads(annotation.read_text())
    if document['schema_version'] != 'omni-interactscope-1.1':
        raise ValueError('Unsupported source schema')
    files = [json.loads(line) for line in (root / 'file_manifest.jsonl').read_text().splitlines() if line.strip()]
    inventory = {f['path']: f for f in files}
    existing = [name for name in inventory if local_asset(root, name).is_file()]
    def verify(name):
        path, spec = local_asset(root, name), inventory[name]
        if config.get('media_verification','sha256')=='decode' and name.startswith('videos/'):
            return name, None if path.stat().st_size else 'empty_media'
        if path.stat().st_size != spec.get('bytes', spec.get('size_bytes', spec.get('size'))):
            return name, 'size_mismatch'
        return name, None if digest(path) == spec['sha256'] else 'sha256_mismatch'
    with ThreadPoolExecutor(max_workers=config['media_workers']) as pool:
        verification = dict(pool.map(verify, existing))
    good = {name for name, error in verification.items() if error is None}
    if 'annotations.json' in inventory and 'annotations.json' not in good:
        raise ValueError('Source annotations fail the official checksum')
    videos = [name for name in good if name.startswith('videos/')]
    public_manifest = root/'external_public_videos.jsonl'
    public_paths = {json.loads(line)['path'] for line in public_manifest.read_text().splitlines() if line.strip()} if public_manifest.exists() else set()
    hosted_expected = set(inventory)-public_paths
    def inspect(name):
        try:
            return name, probe_video(local_asset(root, name))
        except (ValueError, subprocess.SubprocessError, KeyError):
            return name, None
    with ThreadPoolExecutor(max_workers=config['media_workers']) as pool:
        probes = dict(pool.map(inspect, videos))
    tools_path = Path(config['official_code']) / 'configs/tools.yaml'
    import yaml
    tool_definitions = yaml.safe_load(tools_path.read_text())['tools']
    original, converted, runnable, mapping, findings = document['interactions'], [], [], [], []
    timing_anomalies = []
    seen = set()
    for index, row in enumerate(original):
        identifier = row['interaction_id']
        if identifier in seen:
            raise ValueError('Duplicate interaction ID')
        seen.add(identifier)
        reasons = []
        if not row.get('evaluation_splits'):
            reasons.append('official_exclusion')
        try:
            sample = convert(row)
        except (ValueError, KeyError, TypeError) as exc:
            sample = None
            reasons.append('invalid_annotation:' + str(exc))
        if sample is not None:
            if sample['mode'] == 'tool':
                sample['frontend_context']['tool_definitions'] = tool_definitions
            converted.append(sample)
            video = row['video']
            local_asset(root, video)
            if video not in good:
                reasons.append('missing_media' if video not in verification else 'invalid_media:' + verification[video])
            elif probes.get(video) is None:
                reasons.append('undecodable_media')
            else:
                sample['duration_seconds'] = probes[video]['duration']
                if sample['question_time'] >= sample['duration_seconds']:
                    reasons.append('query_after_actual_video_end')
                for ri, reference in enumerate(sample['responses']):
                    if reference['trigger_time'] > sample['duration_seconds']:
                        timing_anomalies.append(dict(interaction_id=identifier,response_index=ri,
                            trigger_time=reference['trigger_time'],actual_duration=sample['duration_seconds'],
                            window_unreachable=reference['window_start']>sample['duration_seconds']))
            if row.get('query_image'):
                reasons.append('unsupported_query_image')
            if row.get('query_type') in ('audio', 'text_audio'):
                # Text-normalized AOI condition, explicitly not original audio attachment input.
                if not row.get('query_text', '').strip():
                    reasons.append('missing_query_transcript')
            conflicts = window_conflicts(sample)
        else:
            conflicts = []
        new_index = None if reasons else len(runnable)
        mapping.append(dict(source_index=index, interaction_id=identifier, sample_index=new_index, reasons=reasons))
        if reasons or conflicts:
            findings.append(dict(source_index=index, interaction_id=identifier, reasons=reasons, window_conflicts=conflicts))
        if not reasons:
            runnable.append(sample)
    source_links = []
    for row in original:
        if row['video'] in good:
            continue
        attr = row['source_attribution']
        source_links.append(dict(video=row['video'], source=attr.get('source'), original_video_url=attr.get('original_video_url'),
                                 source_start_ms=row['source_start_ms'], source_end_ms=row['source_end_ms']))
    links = list({r['video']: r for r in source_links}.values())
    overlap = training_overlap(runnable, config.get('training_metadata'), inventory)
    manifest = dict(version=VERSION, source_dataset=SOURCE, source_annotation=str(annotation),
                    source_sha256=digest(annotation), dataset_revision=config['dataset_revision'],
                    media_verification=config.get('media_verification','sha256'),
                    media_identity_note='Reacquired platform media may differ from original reference bytes; original video size/SHA not checked.' if config.get('media_verification')=='decode' else 'Official reference byte sizes and SHA256 checked.',
                    file_manifest_sha256=digest(root/'file_manifest.jsonl'),
                    hosted_assets_complete=hosted_expected.issubset(good),
                    missing_or_invalid_hosted_assets=sorted(hosted_expected-good),
                    official_code_revision=config['official_code_revision'], tool_definitions_sha256=digest(tools_path),
                    source_qa=len(original), source_targets=sum(len(r['responses']) for r in original),
                    converted_qa=len(converted), runnable_qa=len(runnable), runnable_targets=sum(len(s['responses']) for s in runnable),
                    runnable_tasks=dict(Counter(s['task'] for s in runnable)),
                    official_population_qa=sum(bool(r.get('evaluation_splits')) for r in original),
                    full_coverage=len(runnable) == sum(bool(r.get('evaluation_splits')) for r in original),
                    missing_videos=len(links), verified_videos=len(videos), verification_errors={k:v for k,v in verification.items() if v},
                    decodable_videos=sum(p is not None for p in probes.values()),
                    blocked_reasons=dict(Counter(r for entry in mapping for r in entry['reasons'])),
                    window_conflict_qa=sum(bool(window_conflicts(s)) for s in runnable),
                    timing_anomalies=timing_anomalies,
                    aoi_grid_unreachable_windows=[dict(interaction_id=s['id'],response_indices=unreachable_windows(s))
                                                  for s in runnable if unreachable_windows(s)],
                    training_overlap=overlap, raw_metadata=str(root),
                    input_condition='AOI text request at annotated time; scene audio retained; query audio not consumed; query images unsupported',
                    protocol='AOI one-to-one onset matching, source windows, no first-response gate/dedup/cooldown; not official scores')
    payloads = dict(all_qa=converted, runnable_qa=runnable, source_mapping=mapping, findings=findings,
                    missing_video_sources=links, disjoint_qa=[s for s in runnable if not window_conflicts(s)],
                    no_training_overlap_qa=[s for s in runnable if s['id'] not in overlap['matching_interaction_ids']])
    media = []
    for video in sorted({s['video'] for s in runnable}):
        stat = local_asset(root,video).stat()
        media.append(dict(video=video,size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    payloads['media_manifest'] = media
    encoded = lambda value: (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    manifest['data_sha256'] = hashlib.sha256(encoded(runnable)).hexdigest()
    manifest['source_mapping_sha256'] = hashlib.sha256(encoded(mapping)).hexdigest()
    manifest['media_manifest_sha256'] = hashlib.sha256(encoded(media)).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = output/'manifest.json'
    if previous_manifest.exists() and json.loads(previous_manifest.read_text()) != manifest:
        raise ValueError('Prepared provenance/coverage changed; use a fresh output directory')
    # Prepared files are immutable; media completion requires a fresh output directory.
    for name, value in payloads.items():
        target = output / (name + '.json')
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError('Prepared cohort changed; use a new output directory')
    for name, value in payloads.items():
        write_json(output / (name + '.json'), value)
    # Same source-link export as the author's video-access guide, with one row
    # per public video. This is an acquisition list, never proof of byte identity.
    import csv
    import io
    text = io.StringIO()
    writer = csv.writer(text, delimiter='\t')
    writer.writerow(['video','source','original_video_url','source_start_ms','source_end_ms'])
    seen_public = set()
    for row in original:
        attr = row['source_attribution']
        if attr.get('source') not in ('douyin','xiaohongshu','bilibili') or row['video'] in seen_public:
            continue
        seen_public.add(row['video'])
        url = attr.get('original_video_url','').strip()
        if url.startswith('ttps://'):
            url = 'h'+url
        writer.writerow([row['video'],attr['source'],url,row['source_start_ms'],row['source_end_ms']])
    atomic_write_text(output/'public_video_sources.tsv',text.getvalue())
    assert manifest['data_sha256'] == digest(output/'runnable_qa.json')
    assert manifest['source_mapping_sha256'] == digest(output/'source_mapping.json')
    write_json(output / 'manifest.json', manifest)
    atomic_write_text(output / 'preparation.md', '\n'.join([
        '# Omni-InteractScope under AOI preparation', '',
        f"Source: {manifest['source_qa']} QA / {manifest['source_targets']} targets.",
        f"Runnable: {len(runnable)} QA / {manifest['runnable_targets']} targets; full coverage: {manifest['full_coverage']}.",
        f"Verified videos: {len(videos)}; missing/unverified videos: {len(links)}.",
        f"Task coverage: {manifest['runnable_tasks']}", f"Blocked reasons: {manifest['blocked_reasons']}",
        f"Conflicting-window QA retained: {manifest['window_conflict_qa']}.",
        f"Training overlap check: {overlap['method']}; matching QA: {len(overlap['matching_interaction_ids'])}.",
        'No inference, model export or Judge has run. See manifest.json and missing_video_sources.json.', '']) )
    return manifest


def training_overlap(samples, metadata, inventory):
    """Conservative source-ID/URL and file-hash intersection, not visual dedup."""
    result = dict(checked=False, method='unavailable; no perceptual deduplication', matching_interaction_ids=[])
    if not metadata:
        return result
    path = Path(metadata)
    data = json.loads(path.read_text())
    rows = data['videos'] if isinstance(data, dict) else data
    keys, hashes = set(), set()
    for row in rows:
        for key in ('original_file', 'original_video_url', 'source_url', 'platform_video_id'):
            if row.get(key):
                keys.add(str(row[key])); keys.add(Path(str(row[key])).stem)
        if row.get('sha256'):
            hashes.add(row['sha256'])
    for sample in samples:
        attr = sample['source_sample']['source_attribution']
        candidates = [attr.get('original_video_url'), attr.get('platform_video_id')]
        checksum = inventory.get(sample['video'], {}).get('sha256')
        if any(v and (v in keys or Path(v).stem in keys) for v in candidates) or (checksum and checksum in hashes):
            result['matching_interaction_ids'].append(sample['id'])
    result.update(checked=True, method='exact original source ID/URL or SHA256 intersection; not perceptual deduplication',
                  metadata=str(path.resolve()), metadata_sha256=digest(path))
    return result
