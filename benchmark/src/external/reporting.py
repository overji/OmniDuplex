"""Live common-cohort reporting with separate text and tool-name quality panels."""
from collections import Counter
import json
from pathlib import Path

from benchmark.data.files import atomic_write_text, read_jsonl, write_json
from benchmark.evaluation.timing import match_windows, reference_windows
from benchmark.evaluation.scored_f1 import judged_metrics


def aggregate(samples, predictions, judgments):
    groups = {}
    by_window = {(r['sample_index'],r['response_index']):r for r in judgments}
    for prediction in predictions:
        i = prediction['sample_index']
        sample = samples[i]
        kind = 'tool' if sample['mode']=='tool' else 'text'
        duration = sample['duration_seconds']
        length = 'short' if duration<=120 else 'medium' if duration<=300 else 'long'
        names = [kind, kind+'/task:'+sample['task'], kind+'/length:'+length]
        score = match_windows(sample, prediction['events'])
        rows = [by_window.get((i,ri),dict(status='pending')) for ri,_ in reference_windows(sample)]
        for name in names:
            item = groups.setdefault(name,dict(qa=0,windows=0,outputs=0,hits=0,judgments=[]))
            item['qa'] += 1
            for key, source in [('windows','windows'),('outputs','predictions'),('hits','hits')]:
                item[key] += score[source]
            item['judgments'].extend(rows)
    for name,item in groups.items():
        n,m,h = item['windows'],item['outputs'],item['hits']
        rows = item.pop('judgments')
        item.update(recall=h/n if n else None, precision=h/m if m else 0,
                    f1=2*h/(n+m) if n+m else None, scored_windows=sum(r['status']=='complete' for r in rows),
                    judge_failed=sum(r['status']=='failed' for r in rows))
        item.update(judged_metrics(rows,n,m))
        if name.startswith('tool'):
            quality = item['quality_sum']
            item['tool_name_accuracy_on_matches'] = quality/h if quality is not None and h else None
            item['joint_tool_recall'] = item['scored_recall']
    return groups


def report(path):
    config = json.loads(Path(path).read_text())
    root = Path(config['output'])
    samples = json.loads(Path(config['data']).read_text())
    models, histories, judged = {}, {}, {}
    for job in config['models']:
        label = job['label']
        rows = read_jsonl(root/'runs'/label/'predictions.jsonl', complete_only=True)
        indices = [r['sample_index'] for r in rows]
        if len(indices)!=len(set(indices)):
            raise ValueError('Duplicate prediction index')
        for row in rows:
            i = row['sample_index']
            if type(i) is not int or not 0<=i<len(samples):
                raise ValueError('Invalid prediction index')
            for field in ('video','question','task'):
                if row[field] != samples[i][field]:
                    raise ValueError('Prediction/dataset identity mismatch')
        complete = {r['sample_index']:r for r in rows if r['status']=='complete'}
        models[label] = complete
        histories[label] = Counter(r['status'] for r in rows)
        jp = root/'judge'/label/'current_judgments.json'
        judged[label] = json.loads(jp.read_text()) if jp.exists() else []
    common = set.intersection(*(set(rows) for rows in models.values())) if models else set()
    mismatch = {i for i in common if len({r[i].get('system_prompt_sha256') for r in models.values()})!=1
                or max(r[i]['observed_seconds'] for r in models.values())-min(r[i]['observed_seconds'] for r in models.values())>1e-5}
    common -= mismatch
    paired = {label:aggregate(samples,[rows[i] for i in sorted(common)],judged[label]) for label,rows in models.items()}
    individual = {label:aggregate(samples,list(rows.values()),judged[label]) for label,rows in models.items()}
    coverage = json.loads((Path(config['data']).parent/'manifest.json').read_text())
    summary = dict(protocol=config['protocol'],source_sha256=config['source_sha256'],data_sha256=config['data_sha256'],
                   dataset_revision=config['dataset_revision'], source_qa=coverage['source_qa'],runnable_qa=len(samples),
                   full_coverage=coverage['full_coverage'], paired_qa=len(common), paired_indices=sorted(common),
                   excluded_prompt_duration_mismatches=sorted(mismatch), paired=paired,individual=individual,
                   training_overlap=coverage.get('training_overlap'),
                   statuses={label:dict(value) for label,value in histories.items()},
                   note='AOI adaptation; source windows retained, onset one-to-one matching; no original first-response gate/output merging; text and tools separate')
    lines = ['# Omni-InteractScope under AOI evaluation','',summary['note'],
             f"Official source QA: {coverage['source_qa']}; supported/verified runnable QA: {len(samples)}; full coverage: {coverage['full_coverage']}.",
             f'Paired completed QA: {len(common)}; prompt/duration mismatches excluded: {len(mismatch)}.',
             'Text Judge: '+config['judge']['model']+'. No video Judge. Tool-name sets are scored exactly without instruction references.','',
             '| Model | Complete / failed / missing QA |','|---|---:|']
    for label,rows in models.items():
        status=histories[label]
        lines.append(f"| {label} | {len(rows)} / {sum(v for k,v in status.items() if k!='complete')} / {len(samples)-sum(status.values())} |")
    def fmt(value):
        return '—' if value is None else f'{100*value:.2f}'
    for title,measured in [('Paired common cohort',paired),('Individual coverage (not a paired comparison)',individual)]:
        lines += ['', '## '+title, '', '| Model / group | QA | Recall | Precision | F1 | scored-F1 / tool-weighted F1 | Tool accuracy on timing matches | Judged / windows |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label,groups in measured.items():
            for name,g in groups.items():
                lines.append(f"| {label} / {name} | {g['qa']} | {fmt(g['recall'])} | {fmt(g['precision'])} | {fmt(g['f1'])} | {fmt(g['scored_f1'])} | {fmt(g.get('tool_name_accuracy_on_matches'))} | {g['scored_windows']}/{g['windows']} |")
    lines += ['', 'Official missing videos and unsupported attachments are listed in inputs/preparation.md. '
              'No duplicate outputs are removed. A pending/failed Judge keeps content aggregates undefined, not zero.', '']
    shared = set((coverage.get('training_overlap') or {}).get('matching_interaction_ids', []))
    clean_common = {i for i in common if samples[i]['id'] not in shared}
    summary['paired_without_detected_training_overlap'] = {
        label:aggregate(samples,[rows[i] for i in sorted(clean_common)],judged[label]) for label,rows in models.items()}
    summary['paired_without_detected_training_overlap_qa'] = len(clean_common)
    if coverage.get('training_overlap'):
        lines += ['Training overlap check: '+coverage['training_overlap']['method']+'. '
                  f'Paired QA after excluding detected overlap: {len(clean_common)}. '
                  'Scores are in comparison.json; absence of an exact match does not prove absence of visual overlap.', '']
    write_json(root/'comparison.json',summary)
    atomic_write_text(root/'comparison.md','\n'.join(lines))
    return summary
