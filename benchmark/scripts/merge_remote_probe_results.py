#!/usr/bin/env python3
"""Verify remote split identities and merge into the existing canonical Probe manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.probe.data import SCHEMA, digest
from benchmark.probe.metrics import write_report, record_output


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def read_records(raw):
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Prediction source has an interrupted final line')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def normalize(records, samples, data_hash, canonical_samples, canonical_rows, partition, source):
    by_fingerprint = {}
    for index, sample in enumerate(canonical_samples):
        identity = fingerprint(sample)
        if identity in by_fingerprint:
            raise ValueError('Canonical data has duplicate full-record fingerprints')
        by_fingerprint[identity] = index
    mapping = {}
    for index, sample in enumerate(samples):
        identity = fingerprint(sample)
        if identity not in by_fingerprint:
            raise ValueError('Split QA has no canonical full-record match')
        mapping[index] = by_fingerprint[identity]
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Split contains duplicate QA')
    lookup = {(r['sample_index'], r['response_index'], r['side']): r for r in canonical_rows}
    converted, seen = [], set()
    for prediction in records:
        local_index = prediction['sample_index']
        if local_index not in mapping or prediction.get('status') != 'complete':
            raise ValueError('Out-of-range or incomplete remote prediction')
        if prediction.get('output_protocol') != 'probe_reply_task_text_v1':
            raise ValueError('Unsupported remote Probe output protocol')
        row = lookup.get((mapping[local_index], prediction['response_index'], prediction['side']))
        if row is None:
            raise ValueError('Remote request is absent from the canonical manifest')
        expected = digest([SCHEMA, data_hash, local_index, prediction['response_index'],
                           row['window_start_seconds'] - 1, row['window_end_seconds']])
        if prediction['pair_id'] != expected or prediction['probe_id'] != expected + ':' + row['side']:
            raise ValueError('Remote pair ID or prefix-window identity changed')
        if row['probe_id'] in seen:
            raise ValueError('Duplicate remote prediction')
        seen.add(row['probe_id'])
        if '_merge_source' in prediction:
            raise ValueError('Source is already normalized')
        converted.append({**prediction, 'sample_index': row['sample_index'],
                          'pair_id': row['pair_id'], 'probe_id': row['probe_id'],
                          '_merge_source': {'partition': partition, 'file': str(source),
                                            **{k: prediction[k] for k in ('sample_index', 'pair_id', 'probe_id')}}})
    return converted, mapping


def write_bytes_verified(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Existing artifact differs: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote-dir', type=Path, default=ROOT/'benchmark/results/from_remote_1027')
    p.add_argument('--archive', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--data', type=Path, default=ROOT/'benchmark/results/full_omni_duplex_plus_ab_gemini37_20261021/inputs/all_qa.json')
    p.add_argument('--manifest', type=Path, default=ROOT/'benchmark/probes/full_2110/probe/probe_manifest.jsonl')
    p.add_argument('--first500', type=Path, default=ROOT/'benchmark/exports/aoi_bench_first500.json')
    p.add_argument('--remaining1607', type=Path, default=ROOT/'benchmark/exports/aoi_bench_remaining1607.json')
    args = p.parse_args()
    archive = args.archive or args.remote_dir/'OmniDuplex_remaining1607_probe_predictions_20261027.zip'
    output = args.output_dir or args.remote_dir/'merged_full'
    canonical = json.loads(args.data.read_text()); rows = read_records(args.manifest.read_bytes())
    first = json.loads(args.first500.read_text()); remaining = json.loads(args.remaining1607.read_text())
    if len(first) != 500 or len(remaining) != 1607:
        raise ValueError('Expected exact first500 and remaining1607 exports')
    if len({fingerprint(s) for s in first + remaining}) != 2107:
        raise ValueError('The 500/1607 QA splits overlap or contain duplicates')
    split_manifest = json.loads(args.remaining1607.with_suffix('.manifest.json').read_text())
    if split_manifest['output_sha256'] != sha(args.remaining1607.read_bytes()) or split_manifest['excluded_sha256'] != sha(args.first500.read_bytes()):
        raise ValueError('Export hashes changed')
    public_data = Path(split_manifest['source'])
    if public_data.exists() and (sha(public_data.read_bytes()) != split_manifest['source_sha256'] or json.loads(public_data.read_text()) != first + remaining):
        raise ValueError('Public 2107 QA source changed')
    metadata = {'canonical_data': str(args.data.resolve()), 'canonical_data_sha256': sha(args.data.read_bytes()),
                'canonical_manifest': str(args.manifest.resolve()), 'canonical_manifest_sha256': sha(args.manifest.read_bytes()),
                'public_qa': 2107, 'canonical_qa': len(canonical), 'eligible_requests': len(rows),
                'archive': str(archive.resolve()), 'archive_sha256': sha(archive.read_bytes()),
                'policy': 'Canonical historical runtime indices; original split identifiers retained in _merge_source. Raw text unchanged, no answer recovery. Missing requests remain missing.',
                'models': {}}
    with zipfile.ZipFile(archive) as z:
        packaged = json.loads(z.read('MANIFEST.json'))
        packaged = {r['file']: r for r in packaged['files']}
        for label, short, suffix in [('remote_gemini', 'gemini', 'gemini37'), ('remote_qwen', 'qwen', 'qwen38')]:
            member = suffix + '_remaining1607_probe_predictions_20261027.jsonl'
            raw_remaining = z.read(member); declared = packaged[member]
            if sha(raw_remaining) != declared['sha256'] or len(raw_remaining) != declared['bytes']:
                raise ValueError('Archive file checksum mismatch')
            rem_records = read_records(raw_remaining)
            if len(rem_records) != declared['lines']:
                raise ValueError('Archive record count mismatch')
            raw_first = (args.remote_dir/(short+'_probe_predictions.jsonl')).read_bytes()
            a, map_first = normalize(read_records(raw_first), first, sha(args.first500.read_bytes()), canonical, rows, 'first500', args.remote_dir/(short+'_probe_predictions.jsonl'))
            b, map_remaining = normalize(rem_records, remaining, sha(args.remaining1607.read_bytes()), canonical, rows, 'remaining1607', str(archive)+'!'+member)
            if set(map_first.values()) & set(map_remaining.values()):
                raise ValueError('Canonical split mapping overlaps')
            merged = a + b
            if len({r['probe_id'] for r in merged}) != len(merged):
                raise ValueError('Merged predictions overlap')
            merged.sort(key=lambda r: (r['sample_index'], r['response_index'], r['side']))
            target = output/short; target.mkdir(parents=True, exist_ok=True)
            raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in merged).encode()
            write_bytes_verified(target/'probe_predictions.jsonl', raw)
            write_bytes_verified(args.remote_dir/'remaining1607'/member, raw_remaining)
            scores = write_report(rows, merged, target)
            row_by_id = {r['probe_id']: r for r in rows}
            mismatches = [{'probe_id': r['probe_id'], 'stored': r.get('decision'),
                           'official': record_output(r, row_by_id[r['probe_id']])['decision']}
                          for r in merged if r.get('decision') != record_output(r, row_by_id[r['probe_id']])['decision']]
            (target/'parser_audit.json').write_text(json.dumps({'recovered_answers': 0, 'stored_decision_mismatches': mismatches}, indent=2)+'\n')
            metadata['models'][label] = {'path': str((target/'probe_predictions.jsonl').resolve()),
                                         'sha256': sha(raw), 'first500_requests': len(a),
                                         'remaining1607_requests': len(b), 'merged_requests': len(merged),
                                         'qa_with_predictions': len({r['sample_index'] for r in merged}),
                                         'parser_mismatches': len(mismatches),
                                         'overall': scores['groups']['overall']}
        write_bytes_verified(args.remote_dir/'remaining1607/MANIFEST.json', z.read('MANIFEST.json'))
    mapping_rows = [(part, i, full) for part, mapping in [('first500', map_first), ('remaining1607', map_remaining)] for i, full in mapping.items()]
    with (output/'sample_index_mapping.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['partition', 'split_sample_index', 'canonical_sample_index']); writer.writerows(mapping_rows)
    (output/'merge_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: {field: v[field] for field in ('first500_requests', 'remaining1607_requests', 'merged_requests', 'qa_with_predictions', 'parser_mismatches')} for k, v in metadata['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
