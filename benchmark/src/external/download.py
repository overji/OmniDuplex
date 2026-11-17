"""Pinned, resumable hosted-data download; never start a model or a GPU worker."""
import os
from pathlib import Path
import subprocess
import time

from benchmark.data.files import write_json


def download(config):
    # Keep credential values out of artifacts/logs. Only proxy variables change,
    # and restore the caller's environment when the foreground transfer ends.
    keys = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
            'HF_HUB_DISABLE_XET','HF_HUB_DOWNLOAD_TIMEOUT','HF_HUB_ETAG_TIMEOUT')
    previous = {key: os.environ.pop(key, None) for key in keys}
    try:
        os.environ.update(HF_HUB_DISABLE_XET='1',
                          HF_HUB_DOWNLOAD_TIMEOUT=str(config['download_timeout_seconds']),
                          HF_HUB_ETAG_TIMEOUT=str(config['metadata_timeout_seconds']))
        from huggingface_hub import HfApi, snapshot_download
        endpoint = config['download_endpoint']
        api = HfApi(endpoint=endpoint)
        info = api.dataset_info(config['dataset_repo'], revision=config['dataset_revision'], files_metadata=True)
        if info.sha != config['dataset_revision']:
            raise ValueError('Dataset revision differs')
        target = Path(config['download_root']).resolve()
        for attempt in range(config['download_attempts']):
            try:
                snapshot_download(config['dataset_repo'], repo_type='dataset', revision=info.sha,
                                  endpoint=endpoint, local_dir=target, allow_patterns=['data/**'],
                                  max_workers=config['download_workers'])
                break
            except Exception as exc:
                # Providers may echo signed URLs; do not print exception text.
                print(f'Hosted download attempt {attempt+1}: {type(exc).__name__}; partials retained',flush=True)
                if attempt+1==config['download_attempts']:
                    raise RuntimeError('Hosted download incomplete; resume the same data revision') from None
                time.sleep(1)
        write_json(target / 'download_manifest.json', dict(repo=config['dataset_repo'], revision=info.sha,
                   hosted_bytes=sum(f.size or 0 for f in info.siblings if f.rfilename.startswith('data/')),
                   files=[dict(path=f.rfilename, bytes=f.size) for f in info.siblings if f.rfilename.startswith('data/')]))
        code = Path(config['official_code'])
        if not code.exists():
            subprocess.run(['git', 'clone', config['official_repo'], str(code)], check=True)
            subprocess.run(['git', '-C', str(code), 'checkout', '--detach', config['official_code_revision']], check=True)
        actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != config['official_code_revision']:
            raise ValueError('Official code checkout differs; refusing to change an existing checkout')
    finally:
        for key, value in previous.items():
            os.environ.pop(key,None)
            if value is not None:
                os.environ[key] = value
