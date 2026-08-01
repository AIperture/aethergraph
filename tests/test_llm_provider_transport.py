from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
from pydantic import ValidationError
import pytest

from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateGate,
    ProviderResponseMetadata,
    ProviderRetryExecutor,
    ProviderRetrySettings,
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


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay_s: float) -> None:
        self.sleeps.append(delay_s)
        self.now += delay_s


class _ControlledTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[tuple[float, asyncio.Future[None]]] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay_s: float) -> None:
        released: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.waits.append((delay_s, released))
        await released

    def release(self, index: int) -> None:
        delay_s, released = self.waits[index]
        self.now = max(self.now, delay_s)
        released.set_result(None)


@pytest.mark.asyncio
async def test_retry_executor_honors_provider_minimum_delay_then_succeeds() -> None:
    fake_time = _FakeTime()
    gate = ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep)
    executor = ProviderRetryExecutor(
        rate_gate=gate,
        clock=fake_time.clock,
        random_unit=lambda: 0.0,
    )
    calls = 0

    async def call() -> ProviderCallResult[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMProviderRequestError(
                provider="openai",
                model="gpt-5-nano",
                operation="chat",
                code="provider_rate_limited",
                message="Rate limit reached.",
                retryable=True,
                status_code=429,
                metadata=ProviderResponseMetadata(retry_after_s=0.598),
            )
        return ProviderCallResult(
            "ok",
            ProviderResponseMetadata(request_id="req-success"),
        )

    result = await executor.execute(
        call,
        provider="openai",
        model="gpt-5-nano",
        operation="chat",
    )

    assert result.value == "ok"
    assert calls == 2
    assert fake_time.sleeps == [pytest.approx(0.598)]
    assert [attempt.outcome for attempt in result.attempts] == ["error", "success"]
    assert result.attempts[0].scheduled_delay_s == pytest.approx(0.598)
    assert result.attempts[1].request_id == "req-success"


@pytest.mark.asyncio
async def test_shared_rate_gate_staggers_two_clients_after_the_same_block() -> None:
    controlled_time = _ControlledTime()
    gate = ProviderRateGate(
        clock=controlled_time.clock,
        sleep=controlled_time.sleep,
        random_unit=lambda: 0.5,
        cohort_spread_s=0.05,
    )
    executors = [
        ProviderRetryExecutor(
            rate_gate=gate,
            clock=controlled_time.clock,
            random_unit=lambda: 0.0,
        )
        for _ in range(2)
    ]
    first_attempt_count = 0
    both_started = asyncio.Event()
    calls = [0, 0]

    def provider_call(client_index: int):
        async def call() -> ProviderCallResult[str]:
            nonlocal first_attempt_count
            calls[client_index] += 1
            if calls[client_index] == 1:
                first_attempt_count += 1
                if first_attempt_count == 2:
                    both_started.set()
                await both_started.wait()
                raise LLMProviderRequestError(
                    provider="openai",
                    model="gpt-5-nano",
                    operation="chat",
                    code="provider_rate_limited",
                    message="Rate limit reached.",
                    retryable=True,
                    status_code=429,
                    metadata=ProviderResponseMetadata(retry_after_s=0.5),
                )
            return ProviderCallResult(f"client-{client_index}")

        return call

    tasks = [
        asyncio.create_task(
            executor.execute(
                provider_call(index),
                provider="openai",
                model="gpt-5-nano",
                operation="chat",
            )
        )
        for index, executor in enumerate(executors)
    ]
    for _ in range(10):
        if len(controlled_time.waits) == 2:
            break
        await asyncio.sleep(0)

    assert [delay for delay, _ in controlled_time.waits] == [
        pytest.approx(0.5),
        pytest.approx(0.5375),
    ]
    controlled_time.release(0)
    await asyncio.sleep(0)
    assert calls.count(2) == 1
    controlled_time.release(1)

    results = await asyncio.gather(*tasks)
    assert {result.value for result in results} == {"client-0", "client-1"}
    assert calls == [2, 2]


@pytest.mark.asyncio
async def test_retry_executor_does_not_retry_terminal_quota_error() -> None:
    fake_time = _FakeTime()
    executor = ProviderRetryExecutor(
        rate_gate=ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep),
        clock=fake_time.clock,
    )

    async def call() -> ProviderCallResult[str]:
        raise LLMProviderRequestError(
            provider="openai",
            model="gpt-5-nano",
            operation="chat",
            code="provider_quota_exhausted",
            message="Credits exhausted.",
            retryable=False,
            status_code=429,
        )

    with pytest.raises(LLMProviderRequestError) as captured:
        await executor.execute(
            call,
            provider="openai",
            model="gpt-5-nano",
            operation="chat",
        )

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].retryable is False
    assert fake_time.sleeps == []


@pytest.mark.asyncio
async def test_retry_executor_retries_connect_error_but_not_read_timeout() -> None:
    fake_time = _FakeTime()
    executor = ProviderRetryExecutor(
        rate_gate=ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep),
        clock=fake_time.clock,
        random_unit=lambda: 0.0,
    )
    request = httpx.Request("POST", "https://provider.example/v1/responses")
    connect_calls = 0

    async def connect_then_success() -> ProviderCallResult[str]:
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise httpx.ConnectError("offline", request=request)
        return ProviderCallResult("connected")

    result = await executor.execute(
        connect_then_success,
        provider="ollama",
        model="local-model",
        operation="chat",
    )
    assert result.value == "connected"
    assert connect_calls == 2

    async def read_timeout() -> ProviderCallResult[str]:
        raise httpx.ReadTimeout("late", request=request)

    with pytest.raises(LLMProviderRequestError) as captured:
        await executor.execute(
            read_timeout,
            provider="openai",
            model="gpt-5-nano",
            operation="chat",
        )
    assert captured.value.code == "provider_read_timeout"
    assert captured.value.retryable is False
    assert len(captured.value.attempts) == 1


@pytest.mark.asyncio
async def test_retry_executor_refuses_provider_delay_beyond_policy_budget() -> None:
    fake_time = _FakeTime()
    executor = ProviderRetryExecutor(
        settings=ProviderRetrySettings(max_provider_delay_s=2.0),
        rate_gate=ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep),
        clock=fake_time.clock,
    )

    async def call() -> ProviderCallResult[str]:
        raise LLMProviderRequestError(
            provider="openai",
            model="gpt-5-nano",
            operation="chat",
            code="provider_rate_limited",
            message="Try later.",
            retryable=True,
            metadata=ProviderResponseMetadata(retry_after_s=5.0),
        )

    with pytest.raises(LLMProviderRequestError) as captured:
        await executor.execute(
            call,
            provider="openai",
            model="gpt-5-nano",
            operation="chat",
        )

    assert len(captured.value.attempts) == 1
    assert captured.value.attempts[0].scheduled_delay_s is None
    assert fake_time.sleeps == []


def test_retry_settings_reject_inverted_backoff_bounds() -> None:
    with pytest.raises(ValidationError):
        ProviderRetrySettings(base_delay_s=2.0, max_backoff_s=1.0)


@pytest.mark.asyncio
async def test_generic_chat_recovers_from_temporary_structured_output_429() -> None:
    fake_time = _FakeTime()
    client = GenericLLMClient(provider="openai", model="gpt-5-nano", api_key="test")
    client._provider_retry = ProviderRetryExecutor(
        rate_gate=ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep),
        clock=fake_time.clock,
        random_unit=lambda: 0.0,
    )
    calls = 0

    async def dispatch(messages, **kwargs) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMProviderRequestError(
                provider="openai",
                model="gpt-5-nano",
                operation="chat",
                code="provider_rate_limited",
                message="Rate limit reached.",
                retryable=True,
                status_code=429,
                metadata=ProviderResponseMetadata(retry_after_s=0.598),
            )
        return ProviderCallResult(('{"answer":"ok"}', {"output_tokens": 3}))

    client._chat_dispatch = dispatch  # type: ignore[method-assign]

    text, usage = await client.chat(
        [{"role": "user", "content": "answer"}],
        output_format="json_object",
    )

    assert text == '{"answer": "ok"}'
    assert usage == {"output_tokens": 3}
    assert calls == 2
    assert fake_time.sleeps == [pytest.approx(0.598)]


@pytest.mark.asyncio
async def test_embedding_client_uses_the_same_typed_retry_executor() -> None:
    fake_time = _FakeTime()
    responses = [
        _response(
            429,
            json={"error": {"message": "Busy.", "code": "rate_limit_exceeded"}},
            headers={"retry-after": "0.25"},
        ),
        _response(
            200,
            json={"data": [{"embedding": [0.1, 0.2]}]},
            headers={"x-request-id": "embedding-success"},
        ),
    ]

    class _FakeEmbeddingHttp:
        async def post(self, url, *, headers, json):
            return responses.pop(0)

    client = GenericEmbeddingClient(provider="openai", model="embed-test", api_key="test")
    client._provider_retry = ProviderRetryExecutor(
        rate_gate=ProviderRateGate(clock=fake_time.clock, sleep=fake_time.sleep),
        clock=fake_time.clock,
        random_unit=lambda: 0.0,
    )
    client._client = _FakeEmbeddingHttp()  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    embeddings = await client.embed(["hello"])

    assert embeddings == [[0.1, 0.2]]
    assert responses == []
    assert fake_time.sleeps == [0.5]


@pytest.mark.asyncio
async def test_rate_gate_wait_propagates_cancellation_without_retaining_a_waiter() -> None:
    sleep_started = asyncio.Event()

    async def blocked_sleep(delay_s: float) -> None:
        sleep_started.set()
        await asyncio.Future()

    gate = ProviderRateGate(sleep=blocked_sleep)
    await gate.defer("openai:model", 30.0)
    waiter = asyncio.create_task(gate.wait("openai:model"))
    await sleep_started.wait()
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
