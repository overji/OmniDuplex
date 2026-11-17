#!/usr/bin/env python3
"""Reaggregate existing predictions/Judge JSONL as scored-F1 (Realtime) or scored-Pair (Probe), without API or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmark.data.files import read_jsonl,write_json
from benchmark.data.agent_schema import adapt_sample,run_schema
from benchmark.evaluation import judge,probe_judge
from benchmark.evaluation.scored_f1 import realtime_population, VERSION
from benchmark.evaluation.scored_pair import probe_population, VERSION as PAIR_VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['realtime','probe'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--judgments',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output file; source results are never overwritten')
    samples=json.loads(args.data.read_text())
    filename='predictions.jsonl' if args.mode=='realtime' else 'probe_predictions.jsonl'
    source=args.run/filename;predictions=read_jsonl(source,complete_only=True)
    scored=read_jsonl(args.judgments,complete_only=True)
    if args.mode=='realtime':
        samples=[adapt_sample(s,run_schema(args.run)) for s in samples]
        expected,negatives=judge.prepare_rows(samples,predictions)
        key=lambda row:(row['sample_index'],row['response_index'])
        population=realtime_population(samples,[p for p in predictions if p.get('status')=='complete'])
    else:
        if args.manifest is None:parser.error('Probe requires --manifest')
        manifest=read_jsonl(args.manifest)
        probe_judge.validate_manifest(samples,manifest,hashlib.sha256(args.data.read_bytes()).hexdigest())
        expected,index,counts=probe_judge.prepare_rows(samples,manifest,predictions)
        key=lambda row:row['probe_id']
        population=probe_population(manifest,predictions)
    lookup={key(row):row for row in expected};latest={key(row):row for row in scored}
    if set(latest)-set(lookup):raise ValueError('Judge rows do not belong to this prediction snapshot')
    results=[]
    for identity,row in lookup.items():
        result=latest.get(identity)
        if result is None:
            results.append(dict(row,status='pending'));continue
        for field in ['sample_index','response_index','mode','ability','call_applicable']+(['event_index'] if args.mode=='realtime' else []):
            if row.get(field)!=result.get(field):raise ValueError('Judge/prediction identity mismatch: '+field)
        results.append(result)
    if args.mode=='realtime':
        result=judge.summarize(results,negatives,len(samples),sum(p.get('status')=='complete' for p in predictions),population)
    else:result=probe_judge.summarize(results,counts,len(index),len(manifest),population)
    paths=[args.data,source,args.judgments]+([args.manifest] if args.manifest else [])
    result.update(content_metric=VERSION if args.mode=='realtime' else PAIR_VERSION,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write_json(args.output,result)
    metric = 'scored_f1' if args.mode == 'realtime' else 'scored_pair'
    print(json.dumps(dict(output=str(args.output), **{metric:result['groups'].get('overall',{}).get(metric)})))

if __name__=='__main__':main()
