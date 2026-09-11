"""Minimal OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .video_caption import video_to_data_url


def build_user_content(user_prompt: str, video_path: Path) -> list[dict[str, object]]:
    """Build a multimodal user message containing the video and text evidence."""
    return [
        {
            "type": "video_url",
            "video_url": {"url": video_to_data_url(video_path)},
        },
        {"type": "text", "text": user_prompt},
    ]


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
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_content(user_prompt, video_path),
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"API request failed after {retries + 1} attempt(s): {last_error}"
    )
