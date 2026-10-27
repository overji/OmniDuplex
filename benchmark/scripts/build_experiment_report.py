#!/usr/bin/env python3
"""Recompute comparable experiment tables, preserving raw predictions and source identities."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.probe.data import SCHEMA, digest
from benchmark.probe.metrics import summarize as probe_scores, record_output
from benchmark.evaluation.timing import summarize as realtime_scores

SUITE = ROOT/'benchmark/results/task_text_agent100_20261024_v2/task_specific_probe_512_v3/suite'
FULL = ROOT/'benchmark/results/full_omni_duplex_plus_ab_gemini37_20261021/inputs/all_qa.json'
SUBSET = ROOT/'benchmark/exports/aoi_bench_first500.json'
MANIFEST = ROOT/'benchmark/probes/full_2110/probe/probe_manifest.jsonl'
ABILITIES = {
    'attribute_perception': 'Attribute Perception', 'grounding': 'Temporal Grounding',
    'action_perception': 'Action Perception', 'object_perception': 'Object Perception',
    'emotion_recognition': 'Emotion Recognition', 'event_understanding': 'Event Understanding',
    'text_rich_understanding': 'Text-Rich Understanding', 'object_counting': 'Object Counting',
    'action_counting': 'Action Counting', 'spatial_understanding': 'Spatial Understanding',
    'prospective_reasoning': 'Prospective Reasoning', 'causal_reasoning': 'Causal Reasoning'}
LEVELS = {'Perception': list(ABILITIES)[:5], 'Comprehension': list(ABILITIES)[5:9],
          'Reasoning': list(ABILITIES)[9:]}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def fmt(value):
    return '--' if value is None else f'{100*value:.2f}'


def latex(text):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(text))


class Report:
    def __init__(self, output):
        self.output = output
        self.sources = {}
        self.tables = []
        self.data = {}

    def records(self, path):
        path = Path(path)
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines(keepends=True)
                if line.strip() and line.endswith(b'\n')]
        self.sources[str(path)] = {'sha256_read_snapshot': hashlib.sha256(raw).hexdigest(),
                                   'complete_jsonl_records': len(rows)}
        return rows

    def table(self, name, title, headers, rows, caption):
        rows = [[str(c) for c in row] for row in rows]
        with (self.output/(name+'.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(headers); writer.writerows(rows)
        body = [r'\begin{table}[t]', r'\centering', r'\caption{'+caption+'}',
                r'\label{tab:'+name.replace('_','-')+'}', r'\small',
                r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',
                r'\toprule', ' & '.join(latex(h) for h in headers)+r' \\', r'\midrule']
        body += [' & '.join(latex(c) for c in row)+r' \\' for row in rows]
        body += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}', '']
        (self.output/(name+'.tex')).write_text('\n'.join(body))
        self.tables.append({'name':name, 'title':title, 'headers':headers, 'rows':rows,
                            'caption':caption})


def unique_predictions(rows, key='probe_id'):
    result = {}
    for row in rows:
        if row.get('status') != 'complete' or row[key] in result:
            raise ValueError('Duplicate or incomplete prediction')
        result[row[key]] = row
    return result


def common_probe_scores(rows, models):
    pairs = defaultdict(set)
    for row in rows:
        pairs[row['pair_id']].add(row['probe_id'])
    assert all(len(ids)==2 for ids in pairs.values())
    shared = set.intersection(*[{pair for pair, ids in pairs.items() if ids <= set(predictions)}
                                for predictions in models.values()])
    selected = [r for r in rows if r['pair_id'] in shared]
    wanted = {r['probe_id'] for r in selected}
    if not selected:
        raise ValueError('No shared complete probe pairs')
    return {'pairs':len(shared), 'requests':len(selected),
            'qa':len({r['sample_index'] for r in selected}),
            'metrics':{name:probe_scores(selected,[r for key,r in p.items() if key in wanted])
                       for name,p in models.items()}, 'rows':selected}


def add_probe_table(report, name, title, comparison, labels, note=''):
    rows=[]
    for model,d in comparison['metrics'].items():
        s=d['groups']['overall']
        rows.append([labels[model],*[fmt(s.get(k)) for k in ('positive_accuracy','negative_accuracy',
            'balanced_accuracy','paired_accuracy','output_format_valid_rate','agent_valid_call_rate')]])
    report.table(name,title,['Model','Positive Acc.','Negative Acc.','Balanced Acc.',
        'Paired Acc.','Format Valid','Agent Call Valid'],rows,
        f"{title}. All values are percentages. Models share {comparison['pairs']} fully completed "
        f"pre/post pairs ({comparison['requests']} requests from {comparison['qa']} QA). "
        'Agent call validity is conditional on an Agent reply, not semantic correctness. '+note)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'benchmark/docs/experiment_results_20261027')
    parser.add_argument('--remote-gemini-name',default='Gemini-3.7-flash')
    parser.add_argument('--remote-qwen-name',default='Qwen3.8Omni')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    report=Report(args.output_dir)
    full=json_file(FULL);subset=json_file(SUBSET);manifest=report.records(MANIFEST)
    subset_hash=hashlib.sha256(SUBSET.read_bytes()).hexdigest()
    indexed=defaultdict(list)
    for i,sample in enumerate(full):indexed[fingerprint(sample)].append(i)
    subset_to_full={}
    for i,sample in enumerate(subset):
        candidates=indexed[fingerprint(sample)]
        if len(candidates)!=1:raise ValueError('Subset QA has no unique full-record match')
        subset_to_full[i]=candidates[0]
    full_to_subset={value:key for key,value in subset_to_full.items()}
    with (args.output_dir/'first500_index_mapping.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['first500_index','full2110_index'])
        writer.writerows(subset_to_full.items())
    subset_manifest=[];old_to_new={}
    for original in manifest:
        if original['sample_index'] not in full_to_subset:continue
        si=full_to_subset[original['sample_index']]
        pre=original['window_start_seconds']-1
        pair=digest([SCHEMA,subset_hash,si,original['response_index'],pre,original['window_end_seconds']])
        row={**original,'sample_index':si,'pair_id':pair,'probe_id':pair+':'+original['side']}
        old_to_new[original['probe_id']]=row
        subset_manifest.append(row)
    lookup={r['probe_id']:r for r in subset_manifest}
    labels={'minicpmo':'MiniCPM-o-4.5','qwen3_omni':'Qwen3-Omni-30B-A3B','qwen25_omni':'Qwen2.5-Omni-7B',
            'remote_gemini':args.remote_gemini_name,'remote_qwen':args.remote_qwen_name}
    local={};first500={};coverage=[];parser_mismatches={}
    for name in ('minicpmo','qwen3_omni','qwen25_omni'):
        path=SUITE/'probe'/(name+'_reply_task_text')/'probe_predictions.jsonl'
        local[name]=unique_predictions(report.records(path));converted=[]
        for identity,prediction in local[name].items():
            if identity in old_to_new:
                row=old_to_new[identity]
                converted.append({**prediction,**{k:row[k] for k in ('sample_index','pair_id','probe_id')}})
        first500[name]=unique_predictions(converted)
    for name,filename in [('remote_gemini','gemini_probe_predictions.jsonl'),('remote_qwen','qwen_probe_predictions.jsonl')]:
        records=report.records(ROOT/'benchmark/results/from_remote_1027'/filename)
        for prediction in records:
            row=lookup.get(prediction['probe_id'])
            if row is None or any(prediction[k]!=row[k] for k in ('sample_index','response_index','pair_id','side')):
                raise ValueError('Remote prediction does not match the first500 data identity')
            if prediction.get('output_protocol')!='probe_reply_task_text_v1':
                raise ValueError('Unsupported remote output protocol')
        first500[name]=unique_predictions(records)
    for name,predictions in first500.items():
        pairs=defaultdict(set)
        for row in predictions.values():pairs[row['pair_id']].add(row['side'])
        coverage.append([labels[name],len(predictions),len(subset_manifest),
            sum(sides=={'pre','post'} for sides in pairs.values()),len(subset_manifest)//2,
            len({r['sample_index'] for r in predictions.values()})])
        parser_mismatches[name]=sum(p.get('decision')!=record_output(p,lookup[k])['decision']
                                    for k,p in predictions.items())
    main_probe=common_probe_scores(subset_manifest,first500)
    add_probe_table(report,'probe_first500_common','Probe comparison on the first-500 subset',
        main_probe,labels,'Remote model names are operator-confirmed; exact inference settings were not included with predictions.')
    report.table('probe_first500_coverage','First-500 prediction coverage',
        ['Model','Requests','Expected','Complete Pairs','Expected Pairs','QA'],coverage,
        'Coverage of provided predictions on the first-500 QA export. Missing rows are not assumed to be model errors or correct silence.')
    remote_metrics={name:probe_scores(subset_manifest,list(first500[name].values()))
                    for name in ('remote_gemini','remote_qwen')}
    report.table('probe_remote_supplied','Remote results on all supplied predictions',
        ['Model','Requests','Complete Pairs','Positive Acc.','Negative Acc.','Balanced Acc.',
         'Paired Acc.','Format Valid','Agent Call Valid'],
        [[labels[name],s['completed_probes'],s['complete_pairs'],
          *[fmt(s[k]) for k in ('positive_accuracy','negative_accuracy','balanced_accuracy',
              'paired_accuracy','output_format_valid_rate','agent_valid_call_rate')]]
         for name,d in remote_metrics.items() for s in [d['groups']['overall']]],
        'Scores from all supplied remote predictions, using each model\'s own coverage. '
        'Accuracy denominators contain completed requests; paired accuracy uses only completed pre/post pairs. '
        'Values are percentages. Use the shared-subset table for comparisons with local models.')
    probe_interaction=[]
    for name,d in main_probe['metrics'].items():
        probe_interaction.append([labels[name],*[fmt(d['groups']['mode:'+m]['paired_accuracy']) for m in ('rtd','pr','agent')]])
    report.table('probe_interaction','Probe paired accuracy by interaction',
        ['Model','RTD','PR','Agent'],probe_interaction,
        'Paired accuracy by interaction on the same shared first-500 subset. Values are percentages; both pre and post decisions must be correct.')
    local_probe=common_probe_scores(manifest,local)
    add_probe_table(report,'probe_local_common','Local Probe comparison on the full shared subset',local_probe,labels)
    coverage_full=[]
    for name,predictions in local.items():
        score=probe_scores(manifest,list(predictions.values()))['groups']['overall']
        coverage_full.append([labels[name],len(predictions),len(manifest),fmt(score['positive_accuracy']),
                              fmt(score['negative_accuracy']),fmt(score['paired_accuracy'])])
    report.table('probe_local_coverage','Local full-run snapshots',
        ['Model','Completed','Expected','Positive Acc.','Negative Acc.','Paired Acc.'],coverage_full,
        'Local full-run snapshots with each model\'s own completion coverage. These rows do not share an identical denominator and are descriptive only.')
    qwen_ablation={}
    for variant in ('reply_task_text','no_audio_reply_task_text','no_vision_reply_task_text'):
        key='qwen25_'+variant
        qwen_ablation[key]=unique_predictions(report.records(SUITE/'probe'/('qwen25_omni_'+variant)/'probe_predictions.jsonl'))
    qwen_comparison=common_probe_scores(manifest,qwen_ablation)
    qwen_labels={'qwen25_reply_task_text':'Audio + video','qwen25_no_audio_reply_task_text':'Video only',
                 'qwen25_no_vision_reply_task_text':'Audio only'}
    add_probe_table(report,'qwen25_probe_ablations','Qwen2.5-Omni Probe modality ablations',qwen_comparison,qwen_labels)
    # Compose current Agent results with legacy non-Agent results; never backfill old Agent output.
    rt_sources={
        'minicpmo':[ROOT/'benchmark/results/full_2110_dp6_20261021/realtime/minicpmo',
                    ROOT/'benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/minicpmo_task_text_agent_full'],
        'gander':[ROOT/'benchmark/results/full_2110_dp6_20261021/realtime/gander',
                  ROOT/'benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/gander',
                  ROOT/'benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/gander_task_text_agent_full'],
        'joyai':[ROOT/'benchmark/results/full_2110_phased_8gpu_20261023/suite/realtime/joyai',
                 ROOT/'benchmark/results/task_text_agent100_20261024_v2/continuation/realtime/joyai_task_text_agent_full']}
    rt={}
    for name,runs in rt_sources.items():
        predictions={}
        for source in runs:
            agent='task_text_agent_full' in source.name
            for p in report.records(source/'predictions.jsonl'):
                i=p['sample_index']
                if p.get('status')=='complete' and (full[i]['mode']=='agent')==agent:
                    assert p['video']==full[i]['video'] and p['question']==full[i]['question']
                    predictions[i]=p
        rt[name]=predictions
    common_rt=set.intersection(*(set(p) for p in rt.values()))
    rt_metrics={name:realtime_scores(full,[p[i] for i in sorted(common_rt)]) for name,p in rt.items()}
    rt_labels={'minicpmo':'MiniCPM-o-4.5','gander':'Gander','joyai':'JoyAI-VL-Interaction'}
    report.table('realtime_main','Realtime timing comparison',
        ['Model','QA','Windows','Outputs','Hits','Recall','Precision','F1'],
        [[rt_labels[name],*[s[k] for k in ('samples','windows','predictions','hits')],
          *[fmt(s[k]) for k in ('window_hit_rate','timing_precision','timing_f1')]]
         for name,d in rt_metrics.items() for s in [d['groups']['overall']]],
        f'Realtime timing on {len(common_rt)} shared completed QA. MiniCPM and Gander receive audio-video; JoyAI is vision-only. '
        'Agent uses the task-text rerun; RTD/PR reuse previous runs. No content correctness is required for a timing hit.')
    report.table('realtime_interaction','Realtime timing F1 by interaction',
        ['Model','RTD','PR','Agent'],
        [[rt_labels[name],*[fmt(d['groups']['mode:'+m]['timing_f1']) for m in ('rtd','pr','agent')]]
         for name,d in rt_metrics.items()],
        'Timing F1 by interaction on the same shared QA as the realtime main table. Values are percentages.')
    mini_ablations={'Audio + video':rt['minicpmo']}
    for label,variant in [('Video only','no_audio'),('Audio only','no_vision')]:
        mini_ablations[label]=unique_predictions(report.records(
            SUITE/'realtime'/('minicpmo_'+variant+'_reply_task_text')/'predictions.jsonl'),key='sample_index')
    mini_common=set.intersection(*(set(p) for p in mini_ablations.values()))
    mini_metrics={name:realtime_scores(full,[p[i] for i in sorted(mini_common)]) for name,p in mini_ablations.items()}
    report.table('minicpmo_realtime_ablations','MiniCPM realtime modality ablations',
        ['Input','QA','Recall','Precision','F1','RTD F1','PR F1','Agent F1'],
        [[name,d['evaluated_samples'],*[fmt(d['groups']['overall'][k]) for k in ('window_hit_rate','timing_precision','timing_f1')],
          *[fmt(d['groups']['mode:'+m]['timing_f1']) for m in ('rtd','pr','agent')]] for name,d in mini_metrics.items()],
        f'MiniCPM modality ablations on {len(mini_common)} shared QA. Historical baseline reuse, DP differences and single-seed sampling limit causal interpretation.')
    ability_rows=[]
    for a,title in ABILITIES.items():
        count=next(iter(main_probe['metrics'].values()))['groups']['ability:'+a]['complete_pairs']
        ability_rows.append([title,count,*[fmt(d['groups']['ability:'+a]['paired_accuracy']) for d in main_probe['metrics'].values()]])
    report.table('probe_abilities','Probe paired accuracy across twelve abilities',
        ['Ability','Pairs',*[labels[n] for n in main_probe['metrics']]],ability_rows,
        'Paired decision accuracy by ability on the shared first-500 subset. Percentages; no response-content judge is applied.')
    level_rows=[]
    for level,abilities in LEVELS.items():
        rr=[r for r in main_probe['rows'] if r['ability'] in abilities];ids={r['probe_id'] for r in rr}
        level_rows.append([level,len(rr)//2,*[fmt(probe_scores(rr,[p for k,p in model.items() if k in ids])['groups']['overall']['paired_accuracy'])
                                    for model in first500.values()]])
    report.table('probe_levels','Probe paired accuracy by cognitive level',
        ['Level','Pairs',*[labels[n] for n in main_probe['metrics']]],level_rows,
        'Cognitive levels follow the specified taxonomy: T1--T5 perception, T6--T9 comprehension, and T10--T12 reasoning. Counts are pooled before computing accuracy.')
    content_root=ROOT/'benchmark/results/judge_minicpmo_balanced200_deepseek_flash_20261026'
    content=json_file(content_root/'summary.json');paired=json_file(content_root/'paired_comparison.json')
    content_rows=[]
    for mode in ('realtime','probe'):
        for task in ('rtd','pr','agent'):
            s=content['scores'][mode]['conditional_groups'][task]
            number=lambda v:'--' if v is None else f'{v:.3f}'
            content_rows.append([mode,task.upper(),s['rows'],number(s['response']),number(s['call']),number(s['task'])])
    report.table('content_judge','MiniCPM content judging on balanced 200 QA',
        ['Mode','Task','Replies','Response','Instruction','Task Score'],content_rows,
        'Existing DeepSeek Flash judging on the same balanced 200 QA. Scores range from 0 to 2; Agent averages response and instruction scores. '
        'Realtime uses timing-matched replies; Probe uses positive-prefix true positives. These are different conditional subsets.')
    report.table('content_paired','Content judging on matched reference responses',
        ['Task','Pairs','Realtime Response','Probe Response','Realtime Instruction','Probe Instruction'],
        [[name.upper(),s['n'],*[('--' if s[k] is None else f'{s[k]:.3f}') for k in
          ('realtime_response','probe_response','realtime_call','probe_call')]] for name,s in paired['groups'].items()],
        'Reuse existing independent judgments for the same QA and reference response where both modes replied. '
        'The 90 reference responses come from 65 QA; the Agent subgroup has only seven pairs.')
    # Persist machine-readable metrics without large raw-generation dumps.
    report.data={'snapshot_utc':datetime.now(timezone.utc).isoformat(),
        'remote_model_labels':{'gemini':args.remote_gemini_name,'qwen':args.remote_qwen_name},
        'first500_data_sha256':subset_hash,'first500_qa':len(subset),
        'full_data_sha256':hashlib.sha256(FULL.read_bytes()).hexdigest(),
        'remote_identity_source':'Operator-confirmed names on 2026-10-27; exact API deployment IDs not stored in predictions',
        'first500_eligible_pairs':len(subset_manifest)//2,'first500_eligible_qa':len({r['sample_index'] for r in subset_manifest}),
        'shifted_sample_indices':sum(i!=j for i,j in subset_to_full.items()),
        'parser_decision_mismatches':parser_mismatches,
        'probe_first500':{k:v for k,v in main_probe.items() if k!='rows'},
        'probe_remote_supplied':remote_metrics,
        'probe_local_common':{k:v for k,v in local_probe.items() if k!='rows'},
        'probe_qwen25_ablations':{k:v for k,v in qwen_comparison.items() if k!='rows'},
        'realtime':{'common_qa':len(common_rt),'metrics':rt_metrics},
        'minicpmo_ablations':{'common_qa':len(mini_common),'metrics':mini_metrics},
        'content_judge':content,'content_paired':paired,'tables':report.tables,'sources':report.sources}
    (args.output_dir/'metrics.json').write_text(json.dumps(report.data,ensure_ascii=False,indent=2)+'\n')
    write_documents(report)
    print(json.dumps({'output':str(args.output_dir),'first500_expected_pairs':len(subset_manifest)//2,
        'five_model_shared_pairs':main_probe['pairs'],'local_shared_pairs':local_probe['pairs'],
        'realtime_shared_qa':len(common_rt),'tables':len(report.tables),
        'parser_mismatches':parser_mismatches},ensure_ascii=False,indent=2))


def wide_table(panels, caption, label, best_from=None):
    """Keep panel-specific metrics distinct within one two-column float."""
    lines=[r'\begin{table*}[t]',r'\centering',r'{\small',r'\setlength{\tabcolsep}{3pt}']
    for index,(title,headers,rows) in enumerate(panels):
        if index:lines.append(r'\par\medskip')
        lines += [r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\hline',
                  r'\multicolumn{'+str(len(headers))+r'}{l}{\textbf{'+latex(title)+r'}} \\',
                  r'\hline',' & '.join(r'\textbf{'+latex(h)+'}' for h in headers)+r' \\',r'\hline']
        for row in rows:
            cells=[]
            for column,value in enumerate(row):
                cell=latex(value)
                if best_from is not None and column>=best_from and value!='--':
                    maximum=max(float(r[column]) for r in rows if r[column]!='--')
                    if float(value)==maximum:cell=r'\textbf{'+cell+'}'
                cells.append(cell)
            lines.append(' & '.join(cells)+r' \\')
        lines += [r'\hline',r'\end{tabular}']
    # The font-size group ends before the caption: retain the template defaults.
    lines += ['}',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\end{table*}']
    return '\n'.join(lines)


def reader_main_table(report):
    """Group shared predictions by interaction, cognition and actual video duration."""
    import subprocess
    full=json_file(FULL)
    manifest=report.records(MANIFEST)
    durations={Path(full[r['sample_index']]['video']).name:r['duration_seconds'] for r in manifest}
    rt=report.data['realtime']['metrics']
    for scored in next(iter(rt.values()))['per_sample']:
        video=Path(full[scored['sample_index']]['video']).name
        if video not in durations:
            path=Path('/remote-home/yushengliao/speech/jimodels/datasets/AOI-Bench/videos')/video
            result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                                   '-of','default=noprint_wrappers=1:nokey=1',str(path)],
                                  check=True,capture_output=True,text=True)
            durations[video]=float(result.stdout)
    group_names=['RTD','PR','Agent','Perception','Comprehension','Reasoning',
                 'Short','Medium','Long','Overall']
    def belongs(index,group):
        sample=full[index]
        if group=='Overall':return True
        if group in ('RTD','PR','Agent'):return sample['mode']==group.lower()
        if group in LEVELS:return sample['ability'] in LEVELS[group]
        duration=durations[Path(sample['video']).name]
        return {'Short':duration<=120,'Medium':120<duration<=300,'Long':duration>300}[group]
    with (report.output/'first500_index_mapping.csv').open() as stream:
        mapping={int(r['first500_index']):int(r['full2110_index']) for r in csv.DictReader(stream)}
    rows=[r for r in manifest if r['sample_index'] in set(mapping.values())]
    lookup={(r['sample_index'],r['response_index'],r['side']):r for r in rows}
    models={}
    for name in ('minicpmo','qwen3_omni','qwen25_omni'):
        models[name]=unique_predictions(report.records(SUITE/'probe'/(name+'_reply_task_text')/'probe_predictions.jsonl'))
        known={r['probe_id'] for r in rows}
        models[name]={k:p for k,p in models[name].items() if k in known}
    for name,file in [('remote_gemini','gemini_probe_predictions.jsonl'),('remote_qwen','qwen_probe_predictions.jsonl')]:
        predictions=[]
        for p in report.records(ROOT/'benchmark/results/from_remote_1027'/file):
            row=lookup[(mapping[p['sample_index']],p['response_index'],p['side'])]
            predictions.append({**p,**{k:row[k] for k in ('sample_index','pair_id','probe_id')}})
        models[name]=unique_predictions(predictions)
    shared=common_probe_scores(rows,models)
    assert shared['pairs']==report.data['probe_first500']['pairs']
    labels={'minicpmo':'MiniCPM-o-4.5','qwen3_omni':'Qwen3-Omni-30B-A3B',
            'qwen25_omni':'Qwen2.5-Omni-7B','gander':'Gander','joyai':'JoyAI-VL-Interaction',
            'remote_gemini':report.data['remote_model_labels']['gemini'],
            'remote_qwen':report.data['remote_model_labels']['qwen']}
    probe_groups={};rt_groups={};counts={}
    for group in group_names:
        selected=[r for r in shared['rows'] if belongs(r['sample_index'],group)]
        ids={r['probe_id'] for r in selected}
        probe_groups[group]={name:probe_scores(selected,[p for k,p in predictions.items() if k in ids])['groups']['overall']
                             for name,predictions in models.items()}
        rt_groups[group]={}
        for name,metrics in rt.items():
            selected_rt=[r for r in metrics['per_sample'] if belongs(r['sample_index'],group)]
            sums={k:sum(r[k] for r in selected_rt) for k in ('windows','predictions','hits')}
            recall=sums['hits']/sums['windows'] if sums['windows'] else None
            precision=sums['hits']/sums['predictions'] if sums['predictions'] else 0.
            f1=2*recall*precision/(recall+precision) if recall and precision else 0.
            rt_groups[group][name]={**sums,'samples':len(selected_rt),'recall':recall,'precision':precision,'f1':f1}
        counts[group]={'probe_pairs':len(selected)//2,'probe_qa':len({r['sample_index'] for r in selected}),
                       'realtime_qa':next(iter(rt_groups[group].values()))['samples']}
    for partition in (group_names[:3],group_names[3:6],group_names[6:9]):
        for key in ('probe_pairs','realtime_qa'):
            assert sum(counts[g][key] for g in partition)==counts['Overall'][key]
    lines=[r'\begin{table*}[t]',r'\centering',r'{\small',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{ll*{10}{r}}',r'\hline',
           r' & & \multicolumn{3}{c}{\textbf{Interaction}} & \multicolumn{3}{c}{\textbf{Cognition}} & \multicolumn{3}{c}{\textbf{Video duration}} & \textbf{Overall} \\',
           r'\cline{3-5}\cline{6-8}\cline{9-11}',
           r'\textbf{Model} & \textbf{Metric} & RTD & PR & Agent & Perc. & Compr. & Reason. & Short & Medium & Long & Mean \\',r'\hline']
    for panel,group_scores,metrics in [
        ('Probe',probe_groups,[('Pre','negative_accuracy'),('Post','positive_accuracy'),('Pair','paired_accuracy')]),
        ('Realtime',rt_groups,[('Recall','recall'),('Precision','precision'),('F1','f1')])]:
        names=list(group_scores['Overall'])
        lines += [r'\multicolumn{12}{l}{\textbf{'+panel+r'}} \\',r'\hline']
        for name in names:
            for i,(title,key) in enumerate(metrics):
                cells=[latex(labels[name]) if i==0 else '',title]
                for group in group_names:
                    value=group_scores[group][name][key]
                    displayed=fmt(value)
                    valid=[float(fmt(s[key])) for s in group_scores[group].values() if s[key] is not None]
                    best=value is not None and float(displayed)==max(valid)
                    cells.append(r'\textbf{'+displayed+'}' if best else displayed)
                lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}','}',
        r'\caption{Main results by interaction, cognitive level and video duration (\%). '
        r'Probe Pre/Post are negative/positive decision accuracy; Pair requires both decisions correct. '
        r'Realtime uses one-to-one timing recall, precision and F1. '
        r'Perc./Compr./Reason. group T1--T5/T6--T9/T10--T12, respectively. '
        r'Short: $d\leq120$ s; Medium: $120<d\leq300$ s; Long: $d>300$ s. '
        r'Overall pools all evaluated pairs or timing counts (micro average), rather than averaging overlapping groups. '
        f"Probe shares {shared['pairs']:,} complete pairs across all five models from the first-500 export; "
        f"Realtime shares {report.data['realtime']['common_qa']:,} completed QA across three models. "
        r'The two panels have different metrics and evaluation sets. JoyAI is vision-only. '
        r'\textbf{Bold} indicates the best displayed value within each mode, metric and column, including ties. '
        r'Timing and decision scores do not measure content correctness.}',
        r'\label{tab:main-results}',r'\end{table*}']
    report.data['reader_groups']={'duration_seconds':{'short_max':120,'medium_max':300},
        'counts':counts,'probe':probe_groups,'realtime':rt_groups}
    (report.output/'grouped_main_results.json').write_text(json.dumps(report.data['reader_groups'],ensure_ascii=False,indent=2)+'\n')
    with (report.output/'grouped_main_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Mode','Model','Metric',*group_names])
        for mode,groups,metrics in [('Probe',probe_groups,['negative_accuracy','positive_accuracy','paired_accuracy']),
                                   ('Realtime',rt_groups,['recall','precision','f1'])]:
            for name in groups['Overall']:
                for key in metrics:writer.writerow([mode,labels[name],key,*[fmt(groups[g][name][key]) for g in group_names]])
    return '\n'.join(lines)


def write_documents(report):
    output=report.output
    setup=(r'We evaluate response decisions with independent video-prefix probes and streaming response timing with realtime inference. '
        r'Each reference response yields a negative prefix ending one second before the response window and a positive prefix ending at its end. '
        r'Probe positive/negative accuracy measures whether a model speaks or remains silent; paired accuracy requires both decisions to be correct. '
        r'Unparseable decisions count as incorrect; payload-format validity is reported separately and does not gate a positive decision. '
        r'Missing inference records are reported as missing coverage. '
        r'Realtime recall, precision and F1 use one-to-one onset matching within inclusive annotated windows; timing hits do not require correct content.')
    limitations=(r'The remote model names, Gemini-3.7-flash and Qwen3.8Omni, were confirmed by the operator; '
        r'the two remote files contain predictions but no run configuration or deployment IDs. '
        r'Their pair IDs were verified against the exact first-500 JSON export. '
        r'All 500 QA were uniquely mapped to the local full dataset by full-record fingerprints, correcting 479 shifted sample indices. '
        r'The shared-subset tables control example identity and completion, but do not establish identical remote prompts, '
        r'frame sampling, reasoning budgets or generation limits. Cross-model conclusions remain subject to these differences. '
        r'The first 500 QA are an ordered prefix, not a random representative sample. '
        r'Shared completed subsets may omit the hardest inputs. '
        r'Agent call validity measures output format, not successful tool execution. '
        r'Tools are not executed, and returned tool results are not fed back. '
        r'The cancelled reference-history trial is excluded throughout. '
        r'No new model inference or paid judging was performed to construct this report.')
    probe=report.data['probe_first500']
    scores={name:d['groups']['overall'] for name,d in probe['metrics'].items()}
    gemini=report.data['remote_model_labels']['gemini'];qwen=report.data['remote_model_labels']['qwen']
    probe_analysis=(
        f"On {probe['pairs']:,} shared complete pairs, {latex(gemini)} and {latex(qwen)} achieve "
        f"paired accuracies of {fmt(scores['remote_gemini']['paired_accuracy'])}\\% and "
        f"{fmt(scores['remote_qwen']['paired_accuracy'])}\\%, respectively. "
        r'The local MiniCPM and Qwen3 models respond on more than 90\% of positive prefixes, '
        r'but reject fewer than 15\% of negative prefixes on this subset. '
        r'Their low paired accuracy therefore reflects difficulty withholding premature responses, '
        r'rather than an inability to emit a response. Qwen2.5 exhibits the opposite decision imbalance, '
        r'with low positive accuracy despite high negative accuracy. '
        r'These decision metrics do not establish whether generated response content is correct.')
    rt_analysis=(r'On the 2,008 shared realtime QA, MiniCPM achieves timing F1 of 25.31\%, '
        r'compared with 18.47\% for Gander and 14.86\% for JoyAI. '
        r'Gander has the highest timing precision (38.99\%) but lower recall (12.10\%). '
        r'JoyAI has the highest recall (26.52\%), accompanied by 11,991 outputs and precision of 10.32\%. '
        r'This illustrates why window recall should be interpreted together with output precision. '
        r'JoyAI receives visual input only, so this table is not an input-matched audio-video comparison.')
    ablation_analysis=(r'On 2,107 shared QA, MiniCPM timing F1 decreases from 25.24\% with audio-video '
        r'to 20.63\% with video only and 9.21\% with audio only. '
        r'The overall advantage is not uniform across interactions: Agent timing F1 is 13.80\% '
        r'for video only versus 11.29\% for audio-video. '
        r'Qwen2.5 remains weak in positive decisions under all three input variants; '
        r'its high audio-only negative accuracy should not be read as strong overall decision performance.')
    judge_analysis=(r'Content judging is available only for MiniCPM on a balanced set of 200 QA. '
        r'Comparing the 90 reference responses on which both modes replied gives mean response scores '
        r'of 1.317 for realtime and 1.244 for Probe on a 0--2 scale. '
        r'PR scores are equal at 1.500. The Agent instruction means are 0.000 versus 1.857, '
        r'but this comparison contains only seven Agent pairs and does not demonstrate tool-execution success. '
        r'The previously computed QA-cluster bootstrap interval for the overall response-score difference '
        r'(Probe minus realtime) spans zero; these observations do not establish a reliable overall content ranking.')
    tables={t['name']:t for t in report.tables}
    main_table=reader_main_table(report)
    mini_rows=[['MiniCPM-o-4.5',r[0],*r[2:]] for r in tables['minicpmo_realtime_ablations']['rows']]
    qwen_rows=[['Qwen2.5-Omni-7B',r[0],r[2],r[1],r[4]] for r in tables['qwen25_probe_ablations']['rows']]
    ablation_table=wide_table([
        ('(a) MiniCPM realtime (2,107 shared QA)',
         ['Model','Input','Recall','Prec.','F1','RTD','PR','Agent'],mini_rows),
        (f"(b) Qwen2.5 Probe ({report.data['probe_qwen25_ablations']['pairs']:,} shared pairs)",
         ['Model','Input','Pre','Post','Pair'],qwen_rows)],
        'Modality ablations. All scores are percentages. Realtime RTD/PR/Agent columns report timing F1; '
        'Probe Pre/Post report negative/positive decision accuracy and Pair reports paired accuracy. '
        'Each panel uses its own common completed subset. Best values within each model and metric are bold, including ties. '
        'Historical baseline reuse, DP differences and single-seed sampling limit causal interpretation.',
        'modality-ablations',best_from=2)
    parts=[r'% Generated from existing artifacts; percentages unless specified.',r'\section{Experimental Results}',
        r'\subsection{Evaluation Protocol}',setup,
        r'\subsection{Main Results}',r'Table~\ref{tab:main-results} summarizes Probe and realtime performance.',
        main_table,probe_analysis,rt_analysis,
        r'\subsection{Modality Ablations}',r'Table~\ref{tab:modality-ablations} reports the input-modality ablations.',
        ablation_table,ablation_analysis,
        r'\subsection{Interpretation and Limitations}',limitations]
    (output/'experiments.tex').write_text('\n\n'.join(parts)+'\n')
    (output/'main.tex').write_text('\n'.join([r'\documentclass[10pt,twocolumn]{article}',
        r'\usepackage[margin=0.8in]{geometry}',r'\usepackage{booktabs,graphicx,hyperref}',
        r'\title{AOI-Bench: Experiment Snapshot}',r'\author{}',
        r'\date{'+latex(report.data['snapshot_utc'])+'}',r'\begin{document}',r'\maketitle',
        r'\input{experiments.tex}',r'\end{document}','']))
    lines=['# Benchmark实验结果汇总','',f"快照：{report.data['snapshot_utc']}。",'',
        '可独立编译：`cd benchmark/docs/experiment_results_20261027 && pdflatex main.tex`。',
        '主表、消融附表及正文分析均在`experiments.tex`内，可直接复制或通过\\input引入论文，无需其他表格文件。',
        'LaTeX正文聚焦一张分组主表和一张消融附表；其余覆盖率、细粒度任务和内容judge统计保留在本Markdown及CSV中。',
        '主表逐组数据见`grouped_main_results.csv`，分母及未舍入分数见`grouped_main_results.json`。短/中/长视频按≤120秒、120–300秒（不含120）、>300秒分组。总体为micro汇总；最优值按模式、指标、分组加粗，并列最优同样加粗。',
        '', '## 数据对应和比较范围','',
        '- 两份远端JSONL全部通过前500条导出文件的pair ID校验；模型名由用户确认：'+gemini+'、'+qwen+'。',
        f"- 500条QA中{report.data['first500_eligible_qa']}条有合法Probe配对，共{report.data['first500_eligible_pairs']}对。",
        '- 导出集和旧2110条运行数据的sample_index有479处不同，已按整条QA指纹建立唯一映射；原始文件没有修改。',
        '- 五模型主表仅比较共同完成两侧的配对；另列各模型原始覆盖与本地全量交集，不把这些分母混用。',
        '- 远端模型的API部署ID、prompt、媒体采样和token预算未随文件提供，比较仍受运行配置差异限制。',
        '- 前500条不是随机样本；原始预测的缺失不能直接认定为API失败。',
        '- 以下均沿用官方scorer，没有修复/恢复模型答案；解析决策与保存值不一致数：'+json.dumps(report.data['parser_decision_mismatches']),
        '', '## 主要发现','',
        f"- 五模型共同完成{probe['pairs']}对、{probe['requests']}次请求；{gemini}与{qwen}的成对准确率分别为{fmt(scores['remote_gemini']['paired_accuracy'])}%、{fmt(scores['remote_qwen']['paired_accuracy'])}%。",
        '- MiniCPM和Qwen3的正侧准确率超过90%，负侧不足15%，主要问题是过早开口；Qwen2.5表现为相反的沉默倾向。这里只判断是否开口，不判断内容正确性。',
        '- Realtime共同2008条QA上，MiniCPM/Gander/JoyAI的时间F1为25.31%/18.47%/14.86%。Gander更保守、精确率更高；JoyAI输出更多，召回较高而精确率偏低。JoyAI仅输入视觉。',
        '- MiniCPM在2107条QA上，音视频/仅视觉/仅音频的时间F1为25.24%/20.63%/9.21%；Agent子集并未呈现相同的优势顺序。',
        '- 既有内容judge中，两种模式都回复的90个reference response上，Realtime/Probe回复均分为1.317/1.244（满分2）。QA聚类bootstrap差值区间跨0，不能据此确认总体优劣；Agent调用评分只有7对。',
        '', '## 实验表格','']
    for table in report.tables:
        lines += ['### '+table['title'],'','| '+' | '.join(table['headers'])+' |',
                  '| '+' | '.join('---' for _ in table['headers'])+' |']
        lines += ['| '+' | '.join(row)+' |' for row in table['rows']]
        lines += ['',table['caption'],'',f"LaTeX：`{table['name']}.tex`；CSV：`{table['name']}.csv`。",'']
    lines+=['## 解释边界','',
        '- Probe正侧准确率不是内容正确率；成对准确率要求前后都判对。Realtime时间F1也不验证语义。',
        '- 只有MiniCPM已有平衡200条QA的DeepSeek Flash内容评分；没有对远端模型补调用内容judge。',
        '- 内容均分按不同条件选中的回复计算，需结合90个共同reference response的配对表，不能直接用不同回复集合的均分排名。',
        '- MiniCPM模态消融复用历史音视频基线，未做多随机种子复验；不能据此作严格单因素因果结论。',
        '- LiveStar及尚未完成的其他模型不以旧协议或不完整子集填入主表；本报告仅记录已明确来源的结果。',
        '- 已撤销的GT历史负侧实验完全排除。没有执行新推理或付费API。',
        '', '## 复现','',
        '`python benchmark/scripts/build_experiment_report.py --output-dir benchmark/docs/experiment_results_20261027`',
        '', '可传`--remote-gemini-name`和`--remote-qwen-name`修改展示标签；不改变原始分数。',
        '`metrics.json`保存统计、来源路径及读取快照哈希；`first500_index_mapping.csv`保存索引映射。',
        'LaTeX在当前机器尚未编译（未发现pdflatex/tectonic）；已检查生成表格和输入引用。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
