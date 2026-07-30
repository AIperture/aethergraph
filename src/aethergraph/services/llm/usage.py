from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_llm_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return provider-neutral LLM usage fields without mutating provider usage."""

    raw = dict(usage or {})
    input_tokens = _first_int(raw, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(raw, "completion_tokens", "output_tokens")
    cache_read_tokens = _first_int(
        raw,
        "cache_read_tokens",
        "cache_read_input_tokens",
    )
    cache_write_tokens = _first_int(
        raw,
        "cache_write_tokens",
        "cache_creation_input_tokens",
    )

    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = raw.get(details_key)
        if isinstance(details, dict):
            cache_read_tokens = max(
                cache_read_tokens,
                _int_or_zero(details.get("cached_tokens")),
            )
            cache_write_tokens = max(
                cache_write_tokens,
                _first_int(
                    details,
                    "cache_write_tokens",
                    "cache_creation_tokens",
                    "cache_creation_input_tokens",
                ),
            )

    if "uncached_input_tokens" in raw:
        uncached_input_tokens = _int_or_zero(raw.get("uncached_input_tokens"))
    elif "prompt_tokens" in raw or "input_tokens_details" in raw:
        uncached_input_tokens = max(
            0,
            input_tokens - cache_read_tokens - cache_write_tokens,
        )
    else:
        # Anthropic reports non-cache input separately from cache read/write.
        uncached_input_tokens = input_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "provider_usage_raw": deepcopy(raw),
    }


def normalized_usage_metrics(normalized_usage: dict[str, Any] | None) -> dict[str, int]:
    """Return numeric normalized usage fields suitable for trace metrics."""

    usage = dict(normalized_usage or {})
    return {
        "input_tokens": _int_or_zero(usage.get("input_tokens")),
        "output_tokens": _int_or_zero(usage.get("output_tokens")),
        "cache_read_tokens": _int_or_zero(usage.get("cache_read_tokens")),
        "cache_write_tokens": _int_or_zero(usage.get("cache_write_tokens")),
        "uncached_input_tokens": _int_or_zero(usage.get("uncached_input_tokens")),
    }


def _first_int(raw: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in raw:
            return _int_or_zero(raw.get(key))
    return 0


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = ["normalize_llm_usage", "normalized_usage_metrics"]
