"""Normalize provider token usage and estimate Qwen API charges."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def make_usage_record(*, provider: str, model: str, task: str, video: str,
                      raw_usage: dict[str, Any] | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if raw_usage is None:
        return {
            "schema": "omniduplex.api_usage.v1", "provider": provider, "model": model,
            "task": task, "video": video, "recorded_at": now,
            "status": "unavailable", "reason": "provider response did not include usage",
            "input_tokens": None, "output_tokens": None, "total_tokens": None,
            "cached_input_tokens": None, "estimated_cost_cny": None,
        }

    details = raw_usage.get("prompt_tokens_details") or {}
    input_tokens = _count(raw_usage.get("prompt_tokens"))
    output_tokens = _count(raw_usage.get("completion_tokens"))
    total_tokens = _count(raw_usage.get("total_tokens"))
    cached = _count(details.get("cached_tokens"))
    counts = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens if total_tokens is not None else (
            input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        ),
        "cached_input_tokens": cached,
    }
    record = {
        "schema": "omniduplex.api_usage.v1", "provider": provider, "model": model,
        "task": task, "video": video, "recorded_at": now,
        "status": "measured" if input_tokens is not None and output_tokens is not None else "partial",
        **counts,
        "raw_usage": raw_usage,
        "estimated_cost_cny": None,
    }
    if provider == "qwen" and input_tokens is not None and output_tokens is not None:
        input_rate = float(os.getenv("QWEN_INPUT_PRICE_CNY_PER_MILLION", "0.8"))
        output_rate = float(os.getenv("QWEN_OUTPUT_PRICE_CNY_PER_MILLION", "2.7"))
        cached_rate = float(os.getenv("QWEN_CACHED_INPUT_PRICE_CNY_PER_MILLION", "0.1"))
        cached_billable = min(cached or 0, input_tokens)
        regular_input = input_tokens - cached_billable
        cost = (regular_input * input_rate + cached_billable * cached_rate + output_tokens * output_rate) / 1_000_000
        record["pricing"] = {
            "currency": "CNY", "unit": "per_million_tokens",
            "input": input_rate, "cached_input": cached_rate,
            "output": output_rate, "source": "QWEN_*_PRICE_CNY_PER_MILLION settings or documented defaults",
        }
        record["estimated_cost_cny"] = round(cost, 10)
    return record
