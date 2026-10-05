"""Upload original media to DashScope temporary OSS using its policy API."""

import hashlib
import json
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlsplit


_UPLOAD_CACHE = {}
_FILE_LIFETIME_SECONDS = 48 * 60 * 60


def upload_policy_url(base_url):
    host = urlsplit(base_url).hostname or ""
    if host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}:
        return f"https://{host}/api/v1/uploads"
    if host == "maas.qianwenaiapi.com":
        return "https://dashscope.aliyuncs.com/api/v1/uploads"
    raise ValueError("Set --upload-url or QWEN_UPLOAD_URL to the upload policy endpoint for this region")


def _multipart_chunks(prefix, path, suffix):
    yield prefix
    with path.open("rb") as media:
        while chunk := media.read(1024 * 1024):
            yield chunk
    yield suffix


def upload_video(video_path: Path, *, api_key, model, base_url, timeout, retries, upload_url=None):
    path = video_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"video file does not exist: {path}")
    endpoint = upload_url or upload_policy_url(base_url)
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("upload policy endpoint must be an HTTP(S) URL")
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                 hashlib.sha256(api_key.encode()).hexdigest(), model, base_url, endpoint)
    cached = _UPLOAD_CACHE.get(cache_key)
    if cached and time.monotonic() < cached[1]:
        return cached[0]
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                endpoint + ("&" if "?" in endpoint else "?") + urlencode({"action": "getPolicy", "model": model}),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                policy = json.loads(response.read().decode("utf-8"))["data"]
            limit = float(policy["max_file_size_mb"]) * 1024 * 1024
            if stat.st_size > limit:
                raise ValueError(f"Video exceeds the upload policy's {policy['max_file_size_mb']} MB file limit")
            filename = uuid.uuid4().hex + path.suffix
            key = policy["upload_dir"].rstrip("/") + "/" + filename
            fields = {
                "OSSAccessKeyId": policy["oss_access_key_id"],
                "policy": policy["policy"],
                "Signature": policy["signature"],
                "key": key,
                "x-oss-object-acl": policy["x_oss_object_acl"],
                "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
                "success_action_status": "200",
            }
            boundary = uuid.uuid4().hex
            parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
                     for name, value in fields.items()]
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                         'Content-Type: application/octet-stream\r\n\r\n')
            prefix = "".join(parts).encode("utf-8")
            suffix = f"\r\n--{boundary}--\r\n".encode()
            started = time.monotonic()
            request = urllib.request.Request(
                policy["upload_host"], data=_multipart_chunks(prefix, path, suffix), method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                         "Content-Length": str(len(prefix) + stat.st_size + len(suffix))},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status not in {200, 204}:
                    raise ValueError(f"OSS upload failed with HTTP {response.status}")
                response.read()
            current = path.stat()
            if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
                raise ValueError("Video changed while it was being uploaded; retry with an unchanged source")
            result = "oss://" + key
            _UPLOAD_CACHE[cache_key] = (result, started + _FILE_LIFETIME_SECONDS)
            return result
        except (OSError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Qwen video upload failed after {retries + 1} attempt(s): {last_error}")
