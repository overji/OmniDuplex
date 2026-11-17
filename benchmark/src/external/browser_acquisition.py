"""Foreground normal-browser acquisition; site JS generates requests, no signature forging."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import threading
import urllib.parse
import urllib.request

from benchmark.data.files import append_jsonl, write_json
from benchmark.data.interactscope import local_asset
from benchmark.external.acquisition import acquire_one, actual_platform, canonical_url, failure_reason


def read_browser_request(path):
    """Parse data only. Never execute a copied shell command or reuse its signature."""
    tokens = shlex.split(Path(path).read_text().replace('\\\n',' ').replace('\\\r\n',' '))
    if not tokens or tokens[0]!='curl':
        raise ValueError('Expected a copied curl command')
    headers = {}; endpoint = None
    for i, token in enumerate(tokens):
        if token.startswith(('https://','http://')):
            url = urllib.parse.urlsplit(token)
            if url.path=='/aweme/v1/web/aweme/detail/':endpoint=url
        if token in ('-H','--header'):
            key,_,value=tokens[i+1].partition(':');headers[key.lower()]=value.strip()
        elif token in ('-b','--cookie'):headers['cookie']=tokens[i+1]
        elif token in ('-d','--data','--data-raw','--data-binary','-F','--form'):
            raise ValueError('Only a read-only detail request is accepted')
        elif token in ('-X','--request') and tokens[i+1].upper()!='GET':
            raise ValueError('Only GET is accepted')
    if (not endpoint or endpoint.scheme!='https' or not endpoint.hostname
            or not (endpoint.hostname=='douyin.com' or endpoint.hostname.endswith('.douyin.com'))):
        raise ValueError('Unexpected video-detail endpoint')
    if not headers.get('user-agent') or not headers.get('cookie'):
        raise ValueError('Browser request must include its user agent and literal Cookie header')
    cookies=[]
    for part in headers['cookie'].split(';'):
        name,sep,value=part.strip().partition('=')
        if sep and name:cookies.append(dict(name=name,value=value,domain='.douyin.com',path='/',secure=True,sameSite='Lax'))
    if not cookies:raise ValueError('No cookies parsed')
    return headers['user-agent'],cookies


def download_metadata_video(detail, user_agent, stop):
    def fetch(row,work,settings,cookies=None):
        urls=(detail.get('video') or {}).get('play_addr',{}).get('url_list') or []
        target=work/'source.mp4'
        for url in urls:
            if stop.is_set():raise RuntimeError('download stopped')
            if urllib.parse.urlsplit(url).scheme not in ('http','https'):continue
            request=urllib.request.Request(url,headers={'User-Agent':user_agent,'Referer':'https://www.douyin.com/'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request,timeout=settings['socket_timeout']) as response,target.open('wb') as out:
                    while chunk:=response.read(1024*1024):
                        if stop.is_set():raise RuntimeError('download stopped')
                        out.write(chunk)
                return target
            except Exception:
                if stop.is_set():raise
        raise RuntimeError('media download failed')
    return fetch


async def run(config, request_template, max_items=0, retry_failures_first=False):
    from playwright.async_api import async_playwright
    root=Path(config['dataset_root']);output=Path(config['download_root'])/'public_acquisition'
    output.mkdir(parents=True,exist_ok=True)
    settings=config['public_acquisition'];agent,cookies=read_browser_request(request_template)
    source=json.loads((root/'annotations.json').read_text())['interactions'];unique={}
    for item in source:
        attr=item['source_attribution'];row=dict(video=item['video'],source=attr['source'],
            original_video_url=attr['original_video_url'],source_start_ms=item['source_start_ms'],source_end_ms=item['source_end_ms'])
        if actual_platform(row)=='douyin':unique.setdefault(row['video'],row)
    pending=[r for r in unique.values() if not local_asset(root,r['video']).exists()]
    previous=output/'current.json'
    records={r['video']:r for r in json.loads(previous.read_text()).get('records',[])} if previous.exists() else {}
    if retry_failures_first:
        pending.sort(key=lambda row: records.get(row['video'],{}).get('reason') not in ('Error','RuntimeError'))
    if max_items:pending=pending[:max_items]
    stop=threading.Event();completed=0;inflight=set();pool=ThreadPoolExecutor(max_workers=settings['workers'])
    def save(status):
        write_json(output/'browser_status.json',dict(status=status,pid=os.getpid(),created_at=datetime.now(timezone.utc).isoformat(),
            selected=len(pending),completed_this_run=completed,existing_douyin=len(unique)-len([r for r in unique.values() if not local_asset(root,r['video']).exists()]),
            sha256_checked=False,cookie_values_recorded=False))
    def record(value):
        nonlocal completed
        completed+=1;value['at']=datetime.now(timezone.utc).isoformat();value['request_source']='normal_browser_session'
        records[value['video']]=value;append_jsonl(output/'attempts.jsonl',value)
        write_json(output/'current.json',dict(status='incomplete' if any(r['status'] not in ('complete','existing') for r in records.values()) else 'complete',
            created_at=datetime.now(timezone.utc).isoformat(),sha256_checked=False,
            counts=dict(Counter(r['status'] for r in records.values())),records=list(records.values())))
        save('running');print(json.dumps({k:value.get(k) for k in ('video','status','reason')},ensure_ascii=False),flush=True)
    def acquire(row,detail):
        try:return acquire_one(row,root,settings,fetcher=download_metadata_video(detail,agent,stop))
        except Exception as exc:return dict(row,status='failed',reason=failure_reason(exc),sha256_checked=False)
    async def collect_one():
        done,_=await asyncio.wait(inflight,return_when=asyncio.FIRST_COMPLETED)
        for task in done:inflight.remove(task);record(await task)
    save('starting')
    try:
        async with async_playwright() as p:
            browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            try:
                context=await browser.new_context(user_agent=agent,viewport={'width':1440,'height':900},locale='zh-CN')
                await context.add_cookies(cookies)
                page=await context.new_page();captured=[];session_confirmed=False
                async def capture(response):
                    if '/aweme/v1/web/aweme/detail/' not in response.url:return
                    try:
                        data=await response.json();detail=data.get('aweme_detail')
                        if data.get('status_code')==0 and detail:captured.append(detail)
                    except Exception:pass
                page.on('response',capture)
                for row in pending:
                    while len(inflight)>=settings['workers']:await collect_one()
                    captured.clear()
                    try:
                        await page.goto(canonical_url(row),wait_until='domcontentloaded',timeout=1000*settings['browser_navigation_timeout_seconds'])
                        detail=None
                        for _ in range(settings['browser_metadata_timeout_seconds']):
                            target=re.search(r'/(?:share/)?video/(\d+)',page.url)
                            if target:detail=next((v for v in captured if str(v.get('aweme_id'))==target.group(1)),None)
                            if detail:break
                            await asyncio.sleep(1)
                        if not detail:
                            record(dict(row,status='failed',reason='browser_metadata_unavailable',sha256_checked=False));continue
                        if not session_confirmed:
                            # A successful normal browser request resolves the old
                            # cookie prerequisite; unselected rows are pending,
                            # not falsely claimed as failed download attempts.
                            for previous in records.values():
                                if previous['status']=='blocked' and actual_platform(previous)=='douyin':
                                    previous.update(status='pending',reason='awaiting_browser_acquisition')
                            session_confirmed=True
                        loop=asyncio.get_running_loop()
                        inflight.add(asyncio.ensure_future(loop.run_in_executor(pool,acquire,row,detail)))
                    except Exception as exc:record(dict(row,status='failed',reason=type(exc).__name__,sha256_checked=False))
                while inflight:await collect_one()
            finally:await browser.close()
        save('complete')
    finally:
        stop.set();pool.shutdown(wait=True,cancel_futures=True)
