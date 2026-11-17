#!/usr/bin/env python3
"""Notify on abnormal GRPO exits after observing a live training process."""
import argparse
from email.message import EmailMessage
import fcntl
import glob
import json
import os
from pathlib import Path
import smtplib
import ssl
import time


def secret(env_file):
    value = os.environ.get('MAIL_KEY')
    if value:
        return value
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip().removeprefix('export ').strip() == 'MAIL_KEY':
            return value.strip().strip('"\'')
    raise ValueError('MAIL_KEY not configured')


def process_identity(pid):
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = raw.split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if 'torch.distributed.run' not in command or 'grpo' not in command.lower():
            return None
        return fields[19]
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(tmp, path)


def deliver(config, event):
    key = secret(config['env_file'])
    message = EmailMessage()
    message['From'] = config['sender']
    message['To'] = config['recipient']
    message['Subject'] = 'OmniDuplex GRPO 训练异常停止'
    reason = str(event.get('reason', '训练进程意外退出')).replace(key, '[REDACTED]')
    message.set_content(f"GRPO 训练异常停止。\n\n队列: {event['path']}\n训练进程: {event['pid']}\n原因: {reason}\n发现时间（UTC）: {event['time_utc']}\n\n请查看对应状态文件和训练日志。\n")
    with smtplib.SMTP_SSL(config['smtp_host'], config['smtp_port'], timeout=20, context=ssl.create_default_context()) as client:
        client.login(config['sender'], key)
        client.send_message(message)


def scan(config, state, now):
    """Only arm after seeing a matching live GRPO training launcher."""
    active = state.setdefault('active', {})
    pending = state.setdefault('pending', [])
    sent = state.setdefault('sent', [])
    for filename in glob.glob(config['status_glob']):
        path = Path(filename)
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        candidates = []
        if status.get('stage') == 'training' and status.get('child_pid'):
            candidates.append(('training', status['child_pid']))
        if status.get('baseline') == 'running' and status.get('baseline_pid'):
            candidates.append(('baseline', status['baseline_pid']))
        for kind, pid in candidates:
            identity = process_identity(pid)
            if identity is None:
                continue
            run_id = f'{filename}|{kind}|{pid}|{identity}'
            if run_id not in active and run_id not in sent:
                active[run_id] = dict(path=filename,kind=kind,pid=pid,identity=identity,missing_since=None)
        for run_id, run in list(active.items()):
            if run['path'] != filename:
                continue
            kind = run['kind']
            outcome = status.get('stage') if kind == 'training' else status.get('baseline')
            if outcome == 'complete':
                active.pop(run_id)
                continue
            failure = outcome in ('failed', 'finished_with_errors')
            if failure:
                reason = status.get('error') if kind == 'training' else status.get('baseline_error')
                reason = reason or '训练队列报告失败'
            elif outcome in ('training', 'running'):
                current = process_identity(run['pid'])
                if current == run['identity']:
                    run['missing_since'] = None
                    continue
                if run['missing_since'] is None:
                    run['missing_since'] = now
                    continue
                failure = now - run['missing_since'] >= config['dead_grace_seconds']
                reason = '训练进程退出，但队列尚未报告完成'
            if failure:
                pending.append(dict(id=run_id,path=filename,pid=run['pid'],reason=reason,
                                    time_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(now)),next_attempt=0))
                active.pop(run_id)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    while True:
        now = time.time()
        scan(config, state, now)
        for event in list(state['pending']):
            if now < event['next_attempt']:
                continue
            try:
                deliver(config, event)
            except Exception as exc:
                # Keep credentials out of persisted errors and retry later.
                event['last_error_type'] = type(exc).__name__
                event['next_attempt'] = now + config['retry_seconds']
            else:
                state['sent'].append(event['id'])
                state['pending'].remove(event)
        state['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(now))
        state['watcher_pid'] = os.getpid()
        save(args.state, state)
        time.sleep(config['poll_seconds'])


if __name__ == '__main__':
    main()
