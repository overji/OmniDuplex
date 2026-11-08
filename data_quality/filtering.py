"""Export whole QA whose scored responses all have completed pass judgments."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from benchmark.data.files import read_jsonl,write_json
from benchmark.evaluation.timing import reference_windows,is_scored_response
from data_quality.core import digest


def select_pass_only(samples,name,judgments):
    """Do not drop individual responses or silently add a static-error filter."""
    kept=[];removed=[];mapping=[]
    for index,sample in enumerate(samples):
        refs=[(ri,r) for ri,r in enumerate(sample['responses']) if is_scored_response(sample,r)];reasons=[]
        if not refs:reasons.append(dict(reason='no_scored_response'))
        for ri,_ in refs:
            row=judgments.get(f'{name}:{index}:{ri}')
            reason=('unreviewed' if row is None else 'failed' if row['status']!='complete'
                    else row.get('verdict'))
            if reason!='pass':reasons.append(dict(response_index=ri,reason=reason))
        qa_sha=hashlib.sha256(json.dumps(sample,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        new_index=len(kept) if not reasons else None
        mapping.append(dict(old_sample_index=index,new_sample_index=new_index,qa_sha256=qa_sha,reasons=reasons))
        if reasons:removed.append(dict(original_sample_index=index,reasons=reasons,qa=sample))
        else:kept.append(sample)
    return kept,removed,mapping


def export_pass_only(config,output):
    audit=Path(config['output']);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh filter export directory')
    manifest=json.loads((audit/'manifest.json').read_text());judgments={};files={}
    expected={job['id'] for shard in manifest['shards'] for job in shard}
    for rank in range(len(manifest['shards'])):
        path=audit/f'worker_{rank}/judgments.jsonl'
        files[str(path)]=digest(path)
        for row in read_jsonl(path):
            if row['id'] in judgments:raise ValueError('Duplicate audit result')
            judgments[row['id']]=row
    if set(judgments)!=expected:raise ValueError('Audit does not have complete, unique terminal coverage')
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(policy='all_scored_responses_complete_and_pass',
        scope='benchmark.is_scored_response; Agent delegation replies only',
        static_errors_are_additional_filter=False,audit_manifest=str(audit/'manifest.json'),
        audit_manifest_sha256=digest(audit/'manifest.json'),audit_result_sha256=files,datasets={})
    for name,source in manifest['sources'].items():
        if digest(source['data'])!=source['sha256']:raise ValueError('Audited source changed')
        samples=json.loads(Path(source['data']).read_text());kept,removed,mapping=select_pass_only(samples,name,judgments)
        directory=output/name;directory.mkdir()
        # Byte-exact source backup, plus complete removed payloads and index lineage.
        (directory/'all_qa.original.json').write_bytes(Path(source['data']).read_bytes())
        write_json(directory/'all_qa.json',kept)
        write_json(directory/'removed_qa.json',removed)
        write_json(directory/'index_mapping.json',mapping)
        static={r['sample_index']:r['findings'] for r in manifest['static'] if r['dataset']==name and r['findings']}
        retained_static=[dict(original_sample_index=r['old_sample_index'],new_sample_index=r['new_sample_index'],
            findings=static[r['old_sample_index']]) for r in mapping if r['new_sample_index'] is not None and r['old_sample_index'] in static]
        write_json(directory/'retained_static_findings.json',retained_static)
        reason_counts=Counter(reason for r in removed for reason in {x['reason'] for x in r['reasons']})
        summary['datasets'][name]=dict(original_qa=len(samples),retained_qa=len(kept),removed_qa=len(removed),
            original_sha256=source['sha256'],filtered_sha256=digest(directory/'all_qa.json'),
            retained_scored_responses=sum(len(reference_windows(s)) for s in kept),
            referenced_videos=len({Path(s['video']).name for s in kept}),
            modes=dict(Counter(s['mode'] for s in kept)),removal_reasons=dict(reason_counts),
            retained_qa_with_static_error=sum(any(f['severity']=='error' for f in r['findings']) for r in retained_static))
    write_json(output/'filter_manifest.json',summary)
    print(json.dumps(summary['datasets'],ensure_ascii=False),flush=True)
    return summary
