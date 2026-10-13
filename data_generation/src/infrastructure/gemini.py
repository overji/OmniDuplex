"""Native Gemini video API with run-scoped upload reuse."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import warnings


_session = ContextVar("gemini_video_session", default=None)


def _close(client):
    # Older supported SDK releases do not expose Client.close().
    close = getattr(client, "close", None)
    if close is not None:
        close()


@contextmanager
def video_session():
    resources = {}
    token = _session.set(resources)
    try:
        yield
    finally:
        _session.reset(token)
        for client, uploaded in resources.values():
            try:
                client.files.delete(name=uploaded.name)
            except Exception as exc:
                warnings.warn(f"Gemini temporary file cleanup failed ({type(exc).__name__})", RuntimeWarning)
            finally:
                _close(client)


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini requires google-genai: pip install -r data_generation/requirements-gemini.txt") from exc
    return genai, types


def _active_file(client, uploaded, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = getattr(uploaded.state, "name", None)
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError("Gemini video processing failed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Gemini video processing timed out")
        time.sleep(min(2, remaining))
        uploaded = client.files.get(name=uploaded.name)


def completion(system_prompt, user_prompt, *, video_path, api_key, base_url, model,
               timeout, retries, usage_callback=None, video_url=None):
    if video_url is not None:
        raise ValueError("Gemini uploads the local video; omit --video-url")
    if timeout <= 0:
        raise ValueError("Gemini timeout must be positive")
    genai, types = _sdk()
    resources = _session.get()
    if resources is None:
        with video_session():
            return completion(system_prompt, user_prompt, video_path=video_path,
                              api_key=api_key, base_url=base_url, model=model,
                              timeout=timeout, retries=retries, usage_callback=usage_callback)
    key = (str(video_path.resolve()), api_key, base_url, timeout)
    if key not in resources:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            base_url=base_url, timeout=int(timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=max(1, retries + 1))))
        try:
            uploaded = client.files.upload(file=str(video_path))
        except Exception as exc:
            _close(client)
            raise RuntimeError(f"Gemini video upload failed ({type(exc).__name__})") from exc
        resources[key] = (client, uploaded)
        try:
            uploaded = _active_file(client, uploaded, timeout)
        except Exception as exc:
            raise RuntimeError(f"Gemini video not available: {exc}") from exc
        resources[key] = (client, uploaded)
    client, uploaded = resources[key]
    # Re-check failed/processing uploads on a later stage after an earlier failure.
    uploaded = _active_file(client, uploaded, timeout)
    resources[key] = (client, uploaded)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type), user_prompt],
            config=types.GenerateContentConfig(system_instruction=system_prompt,
                                               temperature=0.1, response_mime_type="application/json"))
        usage = response.usage_metadata
        if usage_callback is not None and usage is not None:
            output = usage.candidates_token_count
            thoughts = getattr(usage, "thoughts_token_count", None) or 0
            usage_callback({
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": output + thoughts if output is not None else None,
                "total_tokens": usage.total_token_count,
                "prompt_tokens_details": {"cached_tokens": usage.cached_content_token_count},
                "gemini_usage_metadata": usage.model_dump(mode="json"),
            })
        candidates = response.candidates or []
        if not candidates or getattr(candidates[0].finish_reason, "name", None) != "STOP" or not response.text:
            raise ValueError("Gemini returned no complete text answer (blocked, truncated, or empty)")
        return response.text
    except Exception as exc:
        detail = str(exc) if isinstance(exc, ValueError) else f"API status {getattr(exc, 'code', 'unavailable')}"
        raise RuntimeError(f"Gemini generation failed ({type(exc).__name__}): {detail}") from exc
