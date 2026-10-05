"""Minimal OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .qwen_upload import upload_video


def build_user_content(user_prompt: str, video_path: Path, *, video_url: str | None = None) -> list[dict[str, object]]:
    """Build a multimodal user message containing the video and text evidence."""
    if video_url is None:
        raise ValueError("A remote video URL is required; use --provider qwen for automatic upload or supply --video-url")
    return [
        {
            "type": "video_url",
            "video_url": {"url": video_url},
        },
        {"type": "text", "text": user_prompt},
    ]


def _stream_content(response, usage_callback=None) -> str:
    """Collect final answer deltas, excluding reasoning and usage events."""
    fragments = []
    finished = False
    for line in response:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            finished = True
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise ValueError(f"API stream error: {chunk['error']}")
        if chunk.get("usage") is not None and usage_callback is not None:
            usage_callback(chunk["usage"])
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason != "stop":
                    raise ValueError(f"API stream did not complete normally: {reason}")
                finished = True
            content = choice.get("delta", {}).get("content")
            if content:
                if not isinstance(content, str):
                    raise ValueError("API stream content must be text")
                fragments.append(content)
    if not finished or not fragments:
        raise ValueError("API stream ended without a complete text answer")
    return "".join(fragments)


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    video_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    retries: int,
    json_array: bool = False,
    provider: str = "openai",
    video_url: str | None = None,
    upload_url: str | None = None,
    usage_callback=None,
) -> str:
    if provider not in {"openai", "qwen"}:
        raise ValueError(f"unsupported provider: {provider}")
    if video_url is not None:
        parsed = urlsplit(video_url)
        valid_oss = provider == "qwen" and parsed.scheme == "oss" and bool(parsed.netloc) and bool(parsed.path)
        valid_remote = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        if not (valid_oss or valid_remote):
            raise ValueError("video_url must be a remote HTTP(S) URL or a Qwen temporary oss:// URL")
    elif provider == "qwen":
        video_url = upload_video(video_path, api_key=api_key, model=model, base_url=base_url,
                                 timeout=timeout, retries=retries, upload_url=upload_url)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_content(user_prompt, video_path, video_url=video_url),
            },
        ],
        "temperature": 0.1,
    }
    if provider == "qwen":
        # Preserve server-side thinking defaults; JSON is enforced by prompts
        # and existing local validation, not a non-thinking-only JSON mode.
        payload.update(stream=True, modalities=["text"], stream_options={"include_usage": True})
    elif not json_array:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if video_url is not None and video_url.startswith("oss://"):
        headers["X-DashScope-OssResourceResolve"] = "enable"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if provider == "qwen":
                    return _stream_content(response, usage_callback)
                body = json.loads(response.read().decode("utf-8"))
            if usage_callback is not None and body.get("usage") is not None:
                usage_callback(body["usage"])
            return body["choices"][0]["message"]["content"]
        except (
            OSError,
            KeyError,
            IndexError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"API request failed after {retries + 1} attempt(s): {last_error}"
    )
