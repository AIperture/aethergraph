from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from aethergraph.services.llm.provider_transport import (
    classify_http_error,
    provider_response_metadata,
)


def _response(
    status_code: int,
    *,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=json,
        headers=headers,
        request=httpx.Request("POST", "https://provider.example/v1/responses"),
    )


def test_openai_temporary_tpm_rate_limit_is_typed_and_preserves_safe_metadata() -> None:
    response = _response(
        429,
        json={
            "error": {
                "message": "Rate limit reached for gpt-5-nano on tokens per min.",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
        headers={
            "retry-after": "0.598",
            "x-request-id": "req_rate_limited",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "34571",
            "x-ratelimit-reset-tokens": "598ms",
            "authorization": "Bearer must-not-escape",
        },
    )

    error = classify_http_error("openai", "gpt-5-nano", "chat", response)

    assert error.code == "provider_rate_limited"
    assert error.retryable is True
    assert error.status_code == 429
    assert error.provider_error_code == "rate_limit_exceeded"
    assert error.provider_error_type == "tokens"
    assert error.metadata.request_id == "req_rate_limited"
    assert error.metadata.retry_after_s == pytest.approx(0.598)
    assert error.metadata.rate_limits[0].resource == "tokens"
    assert error.metadata.rate_limits[0].limit == 200000
    assert error.metadata.rate_limits[0].remaining == 34571
    assert error.metadata.rate_limits[0].reset_after_s == pytest.approx(0.598)
    assert "must-not-escape" not in str(error)


@pytest.mark.parametrize(
    "provider_code",
    ["insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted"],
)
def test_openai_credit_or_spend_429_is_terminal(provider_code: str) -> None:
    response = _response(
        429,
        json={"error": {"message": "Quota exhausted.", "code": provider_code}},
        headers={"retry-after": "1"},
    )

    error = classify_http_error("openai", "gpt-5-nano", "chat", response)

    assert error.code == "provider_quota_exhausted"
    assert error.retryable is False
    assert error.metadata.retry_after_s == 1.0


def test_invalid_provider_request_is_terminal_and_sanitized() -> None:
    response = _response(
        400,
        json={
            "error": {
                "message": "Invalid schema; api_key=sk-secret-value",
                "type": "invalid_request_error",
                "code": "invalid_json_schema",
            }
        },
    )

    error = classify_http_error("openai", "gpt-5-nano", "chat", response)

    assert error.code == "provider_request_rejected"
    assert error.retryable is False
    assert "sk-secret-value" not in str(error)
    assert "[REDACTED]" in str(error)


def test_success_metadata_normalizes_anthropic_absolute_resets() -> None:
    now = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)
    response = _response(
        200,
        headers={
            "request-id": "anthropic-request",
            "anthropic-ratelimit-input-tokens-limit": "100000",
            "anthropic-ratelimit-input-tokens-remaining": "25000",
            "anthropic-ratelimit-input-tokens-reset": "2026-07-31T12:00:02Z",
        },
    )

    metadata = provider_response_metadata("anthropic", response, now=now)

    assert metadata.request_id == "anthropic-request"
    assert metadata.rate_limits[0].resource == "input_tokens"
    assert metadata.rate_limits[0].limit == 100000
    assert metadata.rate_limits[0].remaining == 25000
    assert metadata.rate_limits[0].reset_after_s == 2.0


def test_azure_retry_after_ms_takes_precedence() -> None:
    response = _response(
        429,
        json={"error": {"message": "Too many requests."}},
        headers={
            "retry-after-ms": "750",
            "retry-after": "5",
            "apim-request-id": "azure-request",
            "x-ratelimit-remaining-requests": "0",
        },
    )

    error = classify_http_error("azure", "deployment-a", "chat", response)

    assert error.metadata.retry_after_s == 0.75
    assert error.metadata.request_id == "azure-request"
    assert error.metadata.rate_limits[0].resource == "requests"
    assert error.metadata.rate_limits[0].remaining == 0


def test_gemini_retry_info_body_is_normalized() -> None:
    response = _response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Resource exhausted.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "1.25s",
                    }
                ],
            }
        },
        headers={"x-goog-request-id": "gemini-request"},
    )

    error = classify_http_error("gemini", "gemini-2.5-flash", "chat", response)

    assert error.code == "provider_rate_limited"
    assert error.retryable is True
    assert error.metadata.retry_after_s == 1.25
    assert error.metadata.request_id == "gemini-request"


@pytest.mark.parametrize("provider", ["ollama", "lmstudio", "local"])
def test_local_provider_without_limit_headers_has_no_fabricated_snapshot(provider: str) -> None:
    metadata = provider_response_metadata(provider, _response(200))

    assert metadata.request_id is None
    assert metadata.retry_after_s is None
    assert metadata.rate_limits == ()
