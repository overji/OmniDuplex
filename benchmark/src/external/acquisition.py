"""Foreground public-platform acquisition. No SHA256, no overwrite of existing media."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from benchmark.data.files import append_jsonl, write_json
from benchmark.data.interactscope import local_asset, probe_video


class AcquisitionError(Exception):
    pass


class QuietLogger:
    def debug(self, message): pass
    def warning(self, message): pass
    def error(self, message): pass


def private_cookie_file(source, work):
    """Accept Netscape files or copied Cookie headers; never modify/export values."""
    from http.cookiejar import MozillaCookieJar, Cookie, LoadError
    source = Path(source)
    jar = MozillaCookieJar(str(source))
    try:
        jar.load(ignore_discard=True)
    except LoadError:
        jar = MozillaCookieJar()
        lines = source.read_text().strip().splitlines()
        explicit = [re.sub(r'^cookie\s*:\s*','',line,flags=re.I) for line in lines if re.match(r'^cookie\s*:',line,re.I)]
        header = explicit[0] if explicit else ''.join(line.strip() for line in lines if line.strip() and not line.startswith('#'))
        for part in header.split(';'):
            name,sep,value = part.strip().partition('=')
            if not sep or not re.fullmatch(r'[!#$%&\x27*+.^_`|~0-9A-Za-z-]+',name):
                continue
            jar.set_cookie(Cookie(0,name,value,None,False,'.douyin.com',True,True,'/',True,
                                  True,None,True,None,None,{},False))
    if not len(jar):
        raise ValueError('Cookie file has no usable cookies')
    private = Path(work)/'.cookies.txt'
    # Create privately before writing, rather than chmod after a credential copy.
    descriptor = os.open(private,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.close(descriptor)
    private.chmod(0o600)
    jar.filename = str(private)
    jar.save(ignore_discard=True,ignore_expires=True)
    return private


def canonical_url(row):
    """Repair recorded URL typos, preserving the original in provenance."""
    url = row['original_video_url'].strip()
    if url.startswith('ttps://'):
        url = 'h'+url
    if 'bilibili.com/video/https://' in url:
        match = re.search(r'BV[A-Za-z0-9]{10}',url)
        if match:
            url = 'https://www.bilibili.com/video/'+match.group()+'/'
    return url


def actual_platform(row):
    from urllib.parse import urlparse
    host = urlparse(canonical_url(row)).hostname or ''
    for platform,domains in [('douyin',('douyin.com',)),('xiaohongshu',('xiaohongshu.com','xhslink.com')),
                             ('bilibili',('bilibili.com','b23.tv'))]:
        if any(host==d or host.endswith('.'+d) for d in domains):
            return platform
    return row['source']


def failure_reason(exc):
    message = str(exc).lower()
    if 'cookies' in message or 'sign in' in message or 'login' in message:
        return 'cookies_required'
    if 'geo-restricted' in message:
        return 'source_unavailable_or_geo_restricted'
    if 'unsupported url' in message:
        return 'unsupported_url'
    if '404' in message or 'not found' in message or 'not available' in message:
        return 'source_not_available'
    if '403' in message or '412' in message:
        return 'access_denied'
    if 'timed out' in message or 'timeout' in message:
        return 'timeout'
    return type(exc).__name__


def xhs_media(url, work, timeout, cookies=None):
    """Read only media URLs explicitly supplied by the public page's SSR JSON."""
    import requests
    session = requests.Session()
    session.trust_env = False
    if cookies:
        from http.cookiejar import MozillaCookieJar
        jar = MozillaCookieJar(str(cookies))
        jar.load(ignore_discard=True)
        session.cookies.update(jar)
    headers = {'User-Agent':'Mozilla/5.0','Referer':'https://www.xiaohongshu.com/'}
    response = session.get(url.replace('http://','https://',1),headers=headers,timeout=timeout)
    response.raise_for_status()
    if '/404' in response.url:
        raise AcquisitionError('source not found')
    match = re.search(r'window\.__INITIAL_STATE__\s*=\s*(.*?)(?:</script>)',response.text,re.S)
    if not match:
        raise AcquisitionError('public video metadata unavailable')
    state = re.sub(r'(?<=[[:,])\s*undefined(?=\s*[,}\]])','null',match.group(1))
    try:
        data = json.JSONDecoder().raw_decode(state.strip())[0]
    except ValueError:
        raise AcquisitionError('public video metadata unavailable') from None
    candidates = []
    def visit(value):
        if isinstance(value,dict):
            master = value.get('masterUrl') or value.get('originVideoKey')
            if isinstance(master,str) and master.startswith(('https://','http://')):
                candidates.append(master)
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(data)
    if not candidates:
        raise AcquisitionError('public video URL unavailable; login may be required')
    target = work/'source.mp4'
    with session.get(candidates[0],headers=headers,stream=True,timeout=timeout) as download:
        download.raise_for_status()
        with target.open('wb') as stream:
            for chunk in download.iter_content(1024*1024):stream.write(chunk)
    return target


def fetch_source(row, work, settings, cookies=None, metadata_only=False):
    import yt_dlp
    url = canonical_url(row)
    if actual_platform(row)=='xiaohongshu':
        if metadata_only:return None
        return xhs_media(url,work,settings['socket_timeout'],cookies)
    options = dict(proxy='',quiet=True,no_warnings=True,cachedir=False,logger=QuietLogger(),
                   socket_timeout=settings['socket_timeout'],retries=settings['retries'],extractor_retries=settings['retries'],
                   noplaylist=True,format='bestvideo+bestaudio/best',merge_output_format='mp4',
                   outtmpl=str(work/'source.%(ext)s'),skip_download=metadata_only)
    if cookies:
        # yt-dlp can save updated cookies on close; never mutate the supplied file.
        options['cookiefile']=str(private_cookie_file(cookies,work))
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url,download=not metadata_only)
        if metadata_only:return info
        candidate = Path(downloader.prepare_filename(info))
    if candidate.is_file():return candidate
    files = [p for p in work.glob('source.*') if p.suffix in ('.mp4','.mkv','.webm','.flv','.mov')]
    if len(files)!=1:raise AcquisitionError('download did not produce one merged video')
    return files[0]


def acquire_one(row, root, settings, cookies=None, *, fetcher=None):
    target = local_asset(root,row['video'])
    if target.exists():return dict(row,status='existing',sha256_checked=False)
    temporary = Path(settings['work_dir'])
    temporary.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clip-',dir=temporary) as directory:
        work = Path(directory)
        original = (fetcher or fetch_source)(row,work,settings,cookies)
        before = probe_video(original)
        start, end = row['source_start_ms']/1000., row['source_end_ms']/1000.
        if not 0 <= start < end or before['duration'] < end-settings['duration_tolerance_seconds']:
            raise AcquisitionError('source duration does not cover annotated crop')
        full_source = (settings.get('copy_full_source',False) and start==0 and original.suffix=='.mp4'
                       and abs(before['duration']-end)<=settings.get('full_source_copy_tolerance_seconds',0))
        cropped = original if full_source else work/'clip.mp4'
        if not full_source:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-threads',str(settings['ffmpeg_threads']),
                '-ss',str(start),'-i',str(original),'-t',str(end-start),'-map','0:v:0','-map','0:a:0?',
                '-c:v','libx264','-preset',settings['preset'],'-crf',str(settings['crf']),
                '-threads',str(settings['ffmpeg_threads']),'-c:a','aac','-movflags','+faststart',str(cropped)],
                check=True,capture_output=True,timeout=settings['item_timeout_seconds'])
        after = probe_video(cropped)
        if abs(after['duration']-(end-start))>settings['duration_tolerance_seconds']:
            raise AcquisitionError('crop duration mismatch')
        if before['has_audio'] and not after['has_audio']:
            raise AcquisitionError('scene audio lost during crop')
        target.parent.mkdir(parents=True,exist_ok=True)
        # Atomic, no-clobber publication; temporary and target are on the same filesystem.
        os.link(cropped,target)
        return dict(row,status='complete',sha256_checked=False,bytes=target.stat().st_size,
                    actual_duration_seconds=after['duration'],has_audio=after['has_audio'],
                    media_origin='reacquired_platform_video',encoding=(dict(mode='copy_full_source') if full_source
                        else dict(codec='libx264',preset=settings['preset'],crf=settings['crf'])))


def acquire(config, cookies=None, sources=None):
    root = Path(config['dataset_root']).resolve()
    settings = config['public_acquisition']
    if cookies and not Path(cookies).is_file():
        raise ValueError('Cookie file does not exist')
    output = Path(config['download_root'])/'public_acquisition'
    output.mkdir(parents=True,exist_ok=True)
    annotations = json.loads((root/'annotations.json').read_text())['interactions']
    unique = {}
    for row in annotations:
        attr = row['source_attribution']
        if attr['source'] not in ('douyin','xiaohongshu','bilibili'):continue
        if sources and attr['source'] not in sources:continue
        record = dict(video=row['video'],source=attr['source'],
            original_video_url=attr['original_video_url'],source_start_ms=row['source_start_ms'],source_end_ms=row['source_end_ms'])
        record['detected_platform'] = actual_platform(record)
        record['resolved_source_url'] = canonical_url(record)
        unique.setdefault(row['video'],record)
    records, blocked = [], {}
    # Test the shared cookie prerequisite, rather than making 888 identical
    # unauthenticated failures. These records are marked blocked, not attempted failures.
    douyin = next((r for r in unique.values() if actual_platform(r)=='douyin' and not local_asset(root,r['video']).exists()),None)
    if douyin:
        try:
            with tempfile.TemporaryDirectory(prefix='preflight-',dir=output) as directory:
                fetch_source(douyin,Path(directory),settings,cookies,metadata_only=True)
        except Exception as exc:
            if failure_reason(exc)=='cookies_required':blocked['douyin']='cookies_required'
    pending = []
    for row in unique.values():
        if local_asset(root,row['video']).exists():
            records.append(dict(row,status='existing',sha256_checked=False))
        elif actual_platform(row) in blocked:
            records.append(dict(row,status='blocked',reason=blocked[actual_platform(row)],individual_download_attempted=False,sha256_checked=False))
        else:pending.append(row)
    def process(row):
        try:return acquire_one(row,root,settings,cookies)
        except Exception as exc:return dict(row,status='failed',reason=failure_reason(exc),sha256_checked=False)
    history = output/'attempts.jsonl'
    for row in records:append_jsonl(history,row)
    with ThreadPoolExecutor(max_workers=settings['workers']) as pool:
        for future in as_completed([pool.submit(process,row) for row in pending]):
            record = future.result();record['at']=datetime.now(timezone.utc).isoformat()
            records.append(record);append_jsonl(history,record)
            print(json.dumps({k:record[k] for k in ('video','source','status')},ensure_ascii=False),flush=True)
            write_json(output/'current.json',dict(status='running',finished=len(records),total=len(unique),records=records))
    summary = dict(status='incomplete' if any(r['status'] in ('failed','blocked') for r in records) else 'complete',
                   sha256_checked=False,created_at=datetime.now(timezone.utc).isoformat(),
                   counts=dict(Counter(r['status'] for r in records)),platform_blockers=blocked,records=records)
    write_json(output/'current.json',summary)
    return summary
