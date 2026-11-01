"""OmniPro conversion, comparison, and resumable insertion into an owned queue."""

def execute(args):
    if args.omnipro_command == 'prepare':
        import json
        from benchmark.data.omnipro import prepare
        result = prepare(args.source, args.videos_dir, args.output_dir, args.training_metadata)
        print(json.dumps({k:result[k] for k in ('samples','videos','references','anomalies','training_overlap')}, indent=2))
    elif args.omnipro_command == 'report':
        from benchmark.evaluation.omnipro import report
        report(args.config)
    else:
        from benchmark.runtime.omnipro_queue import run
        run(args.config)
