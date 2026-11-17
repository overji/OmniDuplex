#!/usr/bin/env python3
"""Run in an isolated Playwright environment. Foreground, resumable, no model load."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmark.external.browser_acquisition import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--request-template',type=Path,required=True)
    parser.add_argument('--max-items',type=int,default=0)
    parser.add_argument('--retry-failures-first',action='store_true',
        help='Prioritize prior browser/download errors before missing-metadata items')
    args=parser.parse_args()
    if args.max_items<0:parser.error('max-items must be nonnegative')
    config=json.loads(args.config.read_text())
    os.environ['PLAYWRIGHT_BROWSERS_PATH']=config['public_acquisition']['browser_cache']
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
    asyncio.run(run(config,args.request_template,args.max_items,args.retry_failures_first))
