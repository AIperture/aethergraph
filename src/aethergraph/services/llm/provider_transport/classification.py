"""Central classification and sanitization for provider HTTP responses."""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import json
import math
import re
from typing import Any

import httpx

from .models import (
    LLMProviderRequestError,
    ProviderRateLimitResource,
    ProviderRateLimitSnapshot,
    ProviderResponseMetadata,
)

_DURATION_PART = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)", re.IGNORECASE)
_OPENAI_QUOTA_CODES = {
    "billing_hard_limit_reached",
    "credit_balance_exhausted",
    "insufficient_quota",
    "organization_quota_exceeded",
    "project_quota_exceeded",
}
_SECRET_VALUE = re.compile(
    r"(?i)(api[_ -]?key|authorization|bearer|token|secret)\s*[:=]\s*([^\s,;}]+)"
)
_MAX_MESSAGE_LENGTH = 2_000


def provider_response_metadata(
    provider: str,
    response: httpx.Response,
    *,
    now: datetime | None = None,
) -> ProviderResponseMetadata:
    """
    Normalize allowlisted response metadata without retaining raw headers.

    Examples:
        Read OpenAI token limits:
            ```python
            response = httpx.Response(
                200,
                headers={"x-ratelimit-limit-tokens": "200000"},
            )
            metadata = provider_response_metadata("openai", response)
            assert metadata.rate_limits[0].limit == 200000
            ```

        Keep local providers explicitly unlimited-by-advertisement:
            ```python
            response = httpx.Response(200)
            metadata = provider_response_metadata("ollama", response)
            assert metadata.rate_limits == ()
            ```

    Args:
        provider: Configured provider identifier.
        response: Provider HTTP response whose allowlisted metadata is read.
        now: Optional deterministic UTC wall clock for absolute reset values.

    Returns:
        ProviderResponseMetadata: Sanitized request ID, retry delay, and
        advertised limit snapshots. Missing data remains absent.

    Notes:
        Unknown headers are ignored. An empty rate-limit tuple means the
        provider advertised no usable limits; it does not mean a zero limit.
    """

    normalized_provider = str(provider or "").lower()
    headers = response.headers
    request_id = _first_header(
        headers,
        (
            "x-request-id",
            "request-id",
            "apim-request-id",
            "x-goog-request-id",
            "cf-ray",
        ),
    )
    retry_after_s = _retry_after(headers, now=now)
    prefixes: tuple[str, ...]
    resource_first = False
    if normalized_provider == "anthropic":
        prefixes = ("anthropic-ratelimit-",)
        resource_first = True
    elif normalized_provider in {"azure", "openai", "openrouter", "deepseek"}:
        prefixes = ("x-ratelimit-",)
    else:
        prefixes = ()
    rate_limits = _rate_limit_snapshots(
        headers,
        prefixes=prefixes,
        resource_first=resource_first,
        now=now,
    )
    return ProviderResponseMetadata(
        request_id=request_id,
        retry_after_s=retry_after_s,
        rate_limits=rate_limits,
    )


def classify_http_error(
    provider: str,
    model: str | None,
    operation: str,
    response: httpx.Response,
    *,
    now: datetime | None = None,
) -> LLMProviderRequestError:
    """
    Classify one provider HTTP failure into the canonical LLM error.

    Examples:
        Classify a temporary rate limit:
            ```python
            response = httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
            error = classify_http_error("openai", "gpt-5-nano", "chat", response)
            assert error.retryable is True
            ```

        Classify an invalid request:
            ```python
            response = httpx.Response(400, json={"error": {"message": "Bad schema"}})
            error = classify_http_error("openai", "gpt-5-nano", "chat", response)
            assert error.code == "provider_request_rejected"
            ```

    Args:
        provider: Configured provider identifier.
        model: Provider model or deployment identifier when known.
        operation: Logical provider operation such as chat or embedding.
        response: Non-success provider HTTP response.
        now: Optional deterministic UTC wall clock for absolute reset values.

    Returns:
        LLMProviderRequestError: Sanitized typed failure with retryability and
        normalized response metadata.

    Notes:
        HTTP 429 is not automatically retryable: provider codes that identify
        exhausted credits or spend quotas are terminal. Unknown 429 failures
        remain retryable because they represent the standard HTTP throttle
        semantic unless the provider supplies a terminal quota code.
    """

    payload = _response_payload(response)
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    provider_error_code = _text(error_payload.get("code"))
    provider_error_type = _text(error_payload.get("type") or error_payload.get("status"))
    message = _sanitized_message(
        error_payload.get("message") or payload.get("message") or response.reason_phrase
    )
    metadata = provider_response_metadata(provider, response, now=now)
    body_delay = _body_retry_after_s(payload)
    if metadata.retry_after_s is None and body_delay is not None:
        metadata = ProviderResponseMetadata(
            request_id=metadata.request_id,
            retry_after_s=body_delay,
            rate_limits=metadata.rate_limits,
        )

    status_code = response.status_code
    normalized_code = (provider_error_code or "").lower()
    if status_code == 429 and normalized_code in _OPENAI_QUOTA_CODES:
        code = "provider_quota_exhausted"
        retryable = False
    elif status_code == 429:
        code = "provider_rate_limited"
        retryable = True
    elif status_code in {408, 409, 425}:
        code = "provider_request_temporarily_unavailable"
        retryable = True
    elif status_code in {500, 502, 503, 504}:
        code = "provider_unavailable"
        retryable = False
    else:
        code = "provider_request_rejected"
        retryable = False

    return LLMProviderRequestError(
        provider=provider,
        model=model,
        operation=operation,
        code=code,
        message=message,
        retryable=retryable,
        status_code=status_code,
        provider_error_code=provider_error_code,
        provider_error_type=provider_error_type,
        metadata=metadata,
    )


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {"message": response.text}
    return payload if isinstance(payload, dict) else {"message": str(payload)}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitized_message(value: Any) -> str:
    message = " ".join(str(value or "Provider request failed.").split())
    message = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}=[REDACTED]", message)
    return message[:_MAX_MESSAGE_LENGTH]


def _first_header(headers: httpx.Headers, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = headers.get(name)
        if value:
            return value.strip()[:256]
    return None


def _retry_after(headers: httpx.Headers, *, now: datetime | None) -> float | None:
    retry_after_ms = _finite_float(headers.get("retry-after-ms"))
    if retry_after_ms is not None:
        return max(0.0, retry_after_ms / 1_000.0)
    value = headers.get("retry-after")
    if not value:
        return None
    duration = _duration_seconds(value)
    if duration is not None:
        return duration
    return _absolute_delay_seconds(value, now=now)


def _rate_limit_snapshots(
    headers: httpx.Headers,
    *,
    prefixes: tuple[str, ...],
    resource_first: bool,
    now: datetime | None,
) -> tuple[ProviderRateLimitSnapshot, ...]:
    snapshots: list[ProviderRateLimitSnapshot] = []
    resource_names: tuple[tuple[ProviderRateLimitResource, str], ...] = (
        ("requests", "requests"),
        ("tokens", "tokens"),
        ("input_tokens", "input-tokens"),
        ("output_tokens", "output-tokens"),
    )
    for prefix in prefixes:
        for resource, header_resource in resource_names:
            if resource_first:
                limit_name = f"{prefix}{header_resource}-limit"
                remaining_name = f"{prefix}{header_resource}-remaining"
                reset_name = f"{prefix}{header_resource}-reset"
            else:
                limit_name = f"{prefix}limit-{header_resource}"
                remaining_name = f"{prefix}remaining-{header_resource}"
                reset_name = f"{prefix}reset-{header_resource}"
            limit = _finite_int(headers.get(limit_name))
            remaining = _finite_int(headers.get(remaining_name))
            reset_value = headers.get(reset_name)
            reset_after_s = _duration_seconds(reset_value)
            if reset_after_s is None and reset_value:
                reset_after_s = _absolute_delay_seconds(reset_value, now=now)
            if limit is None and remaining is None and reset_after_s is None:
                continue
            snapshots.append(
                ProviderRateLimitSnapshot(
                    resource=resource,
                    limit=limit,
                    remaining=remaining,
                    reset_after_s=reset_after_s,
                )
            )
    return tuple(snapshots)


def _duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    numeric = _finite_float(normalized)
    if numeric is not None:
        return max(0.0, numeric)
    matches = list(_DURATION_PART.finditer(normalized))
    if not matches or "".join(match.group(0) for match in matches) != normalized:
        return None
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3_600.0}
    return sum(float(match.group("value")) * multipliers[match.group("unit")] for match in matches)


def _absolute_delay_seconds(value: str, *, now: datetime | None) -> float | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return max(0.0, (parsed.astimezone(UTC) - current.astimezone(UTC)).total_seconds())


def _finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_int(value: str | None) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0:
        return None
    return int(parsed)


def _body_retry_after_s(payload: dict[str, Any]) -> float | None:
    details = payload.get("details")
    if not isinstance(details, list):
        error = payload.get("error")
        details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        retry_delay = detail.get("retryDelay") or detail.get("retry_delay")
        parsed = _duration_seconds(_text(retry_delay))
        if parsed is not None:
            return parsed
    return None
