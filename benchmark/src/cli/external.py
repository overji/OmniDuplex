"""External AOI experiments have explicit prepare/run/report lifecycle stages."""
import json


def execute(args):
    if args.external_command == 'prepare':
        from benchmark.external.preparation import prepare_suite
        result = prepare_suite(args.config,args.output_dir,download=args.download,public_videos=args.public_videos,
                               skip_media_sha256=args.skip_media_sha256)
    elif args.external_command == 'acquire':
        from benchmark.external.acquisition import acquire
        from benchmark.external.preparation import read_config
        config = read_config(args.config)
        if args.browser_request:
            import subprocess
            from pathlib import Path
            if args.sources and set(args.sources)!={'douyin'}:
                raise ValueError('Browser mode currently supports Douyin only')
            if args.max_items<0:raise ValueError('max-items must be nonnegative')
            python=args.browser_python or config['public_acquisition']['browser_python']
            script=Path(__file__).resolve().parents[2]/'scripts/acquire_interactscope_browser.py'
            subprocess.run([str(python),str(script),'--config',str(args.config.resolve()),
                '--request-template',str(args.browser_request.resolve()),'--max-items',str(args.max_items)],check=True)
            result = json.loads((Path(config['download_root'])/'public_acquisition/browser_status.json').read_text())
        else:
            result = acquire(config,args.cookies,args.sources)
        result = {k:v for k,v in result.items() if k!='records'}
    elif args.external_command == 'report':
        from benchmark.external.runtime import validate
        from benchmark.external.reporting import report
        config = json.loads(args.config.read_text())
        validate(config)
        result = report(args.config)
    else:
        from benchmark.external.runtime import run
        result = run(args.config,args.model,args.gpus,stage=args.stage,
                     judge_base_url=args.judge_base_url,check_only=args.dry_run)
    if args.external_command == 'prepare':
        result = {k:result[k] for k in ('source_qa','runnable_qa','runnable_targets','full_coverage','blocked_reasons')}
    print(json.dumps(result,ensure_ascii=False,indent=2))
