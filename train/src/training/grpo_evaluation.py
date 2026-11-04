"""Fixed held-out, autonomous rollouts for comparable GRPO timing diagnostics."""
import json
from pathlib import Path

import torch
from benchmark.data.files import append_jsonl, atomic_write_text, write_json
from benchmark.data.timestamps import seconds
from benchmark.evaluation.timing import match_windows
from train.src.domain.agent import training_sample
from train.src.infrastructure.distributed import rng_state, restore_rng
from train.src.infrastructure.media import Clip
from train.src.models.streaming import rollout


def summarize(rows):
    """Micro timing metrics; empty/native control attempts are reported separately."""
    good = [r for r in rows if r.get('status') == 'complete']
    tp = sum(r['tp'] for r in good)
    fp = sum(r['fp'] for r in good)
    fn = sum(r['fn'] for r in good)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(scored=len(good), failed=len(rows)-len(good), tp=tp, fp=fp, fn=fn,
                fp_per_qa=fp / len(good) if good else None,
                unparsed_attempts=sum(r['unparsed_attempts'] for r in good),
                precision=precision, recall=recall, f1=2*tp/max(2*tp+fp+fn, 1))


def comparison(baseline, current):
    """A failed or missing video cannot appear as an FP improvement."""
    base = {r['sample_index']: r for r in baseline if r.get('status') == 'complete'}
    now = {r['sample_index']: r for r in current if r.get('status') == 'complete'}
    shared = sorted(base.keys() & now.keys())
    if not shared:
        return dict(paired_qa=0, baseline=None, current=None, delta=None)
    before, after = summarize([base[i] for i in shared]), summarize([now[i] for i in shared])
    keys = ('fp', 'fp_per_qa', 'precision', 'recall', 'f1')
    return dict(paired_qa=len(shared), baseline=before, current=after,
                delta={key: after[key]-before[key] for key in keys})


def evaluate(engine, model, duplex, args, config, rank, world):
    """Each rank evaluates different QA; ZeRO-2 parameters are already replicated.

    Fixed sample seeds and unchanged streaming decode make before/after results
    comparable. References are used only after autonomous generation for scoring.
    This does not run a content judge and does not contribute training gradients.
    """
    settings = config['grpo_evaluation']
    selected = settings['sample_indices']
    validation = json.loads((args.prepared / 'validation.json').read_text())
    lookup = {r['sample_index']: r for r in validation}
    if not selected or len(selected) != len(set(selected)) or any(i not in lookup for i in selected):
        raise ValueError('GRPO evaluation must use unique prepared validation sample indices')
    records = [lookup[i] for i in selected]
    state, was_training = rng_state(), engine.training
    rows = []
    try:
        engine.eval()
        with torch.no_grad():
            for record in records[rank::world]:
                index = record['sample_index']
                try:
                    clip = Clip(record, config, None)
                    trajectory = rollout(model, duplex, record, clip, config, settings['seed'] + index)
                    sample = training_sample(record['sample'])
                    matched = match_windows(sample, trajectory['events'])
                    starts = sum(c['new_response'] for second, c in enumerate(trajectory['chunks'], 1)
                                 if second >= seconds(sample['question_time']) + 1)
                    rows.append(dict(sample_index=index, status='complete', mode=sample['mode'],
                        tp=matched['hits'], fp=len(trajectory['events'])-matched['hits'],
                        fn=matched['windows']-matched['hits'],
                        unparsed_attempts=max(0, starts-len(trajectory['events'])),
                        events=trajectory['events']))
                except Exception as exc:
                    rows.append(dict(sample_index=index, status='failed', error_type=type(exc).__name__, error=str(exc)))
                print(json.dumps(dict(stage='grpo_evaluation_sample', step=engine.global_steps,
                                      rank=rank, sample_index=index, status=rows[-1]['status'])), flush=True)
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, rows)
        if rank == 0:
            rows = sorted((r for part in gathered for r in part), key=lambda r: r['sample_index'])
            directory = args.output / 'timing_evaluation'
            directory.mkdir(exist_ok=True)
            step = engine.global_steps
            write_json(directory / f'step-{step}.json', dict(step=step, summary=summarize(rows), rows=rows))
            baseline_path = directory / 'step-0.json'
            baseline = json.loads(baseline_path.read_text())['rows']
            report = dict(step=step, **comparison(baseline, rows))
            append_jsonl(directory / 'history.jsonl', report)
            write_json(directory / 'latest.json', report)
            lines = ['# GRPO held-out timing check', '',
                'Fixed validation QA and seeds; timing only, no content judge. FP counts parsed response events.', '',
                '| Step | Paired QA | FP | FP/QA | Precision | Recall | F1 | ΔFP |',
                '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for line in (directory / 'history.jsonl').read_text().splitlines():
                item = json.loads(line)
                score, delta = item['current'], item['delta']
                if score is not None:
                    lines.append(f"| {item['step']} | {item['paired_qa']} | {score['fp']} | {score['fp_per_qa']:.2f} | "
                        f"{score['precision']:.2%} | {score['recall']:.2%} | {score['f1']:.2%} | {delta['fp']:+d} |")
            atomic_write_text(directory / 'report.md', '\n'.join(lines)+'\n')
            print(json.dumps(dict(stage='grpo_evaluation_complete', **report)), flush=True)
        torch.distributed.barrier()
    finally:
        restore_rng(state)
        engine.train(was_training)
