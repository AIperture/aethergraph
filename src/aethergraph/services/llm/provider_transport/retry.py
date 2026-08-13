"""Bounded retry execution for provider transport attempts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import random
import time
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx

from .classification import classify_transport_error
from .models import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRetrySettings,
    ProviderTransportAttempt,
)
from .rate_gate import ProviderRateGate, ProviderRateGateDeadlineExceededError

ResultT = TypeVar("ResultT")
Clock = Callable[[], float]
RandomUnit = Callable[[], float]


class ProviderRetryExecutor:
    """Execute one logical provider call under a bounded retry policy."""

    def __init__(
        self,
        settings: ProviderRetrySettings | None = None,
        *,
        rate_gate: ProviderRateGate | None = None,
        base_url: str | None = None,
        credential: str | None = None,
        clock: Clock = time.monotonic,
        random_unit: RandomUnit = random.random,
    ) -> None:
        """
        Construct a retry executor with injectable deterministic dependencies.

        Examples:
            Use the default bounded policy:
                ```python
                executor = ProviderRetryExecutor()
                ```

            Share a container gate across clients:
                ```python
                gate = ProviderRateGate()
                executor = ProviderRetryExecutor(rate_gate=gate)
                ```

        Args:
            settings: Validated retry limits and backoff parameters.
            rate_gate: Shared container-local provider rate gate.
            base_url: Resolved provider endpoint used to isolate quota domains.
            credential: Resolved provider credential, retained only as a
                non-reversible in-memory fingerprint.
            clock: Monotonic clock used for elapsed and deadline budgets.
            random_unit: Random value supplier in the inclusive range zero to
                one for bounded positive jitter.

        Returns:
            None: Initializes the executor.

        Notes:
            The executor retries only failures explicitly classified as safe.
            Provider adapters remain single-attempt functions.
        """

        self.settings = settings or ProviderRetrySettings()
        self.rate_gate = rate_gate or ProviderRateGate(clock=clock)
        self._rate_limit_scope = _rate_limit_scope(base_url, credential)
        self._clock = clock
        self._random_unit = random_unit

    async def execute(
        self,
        call: Callable[[], Awaitable[ProviderCallResult[ResultT]]],
        *,
        provider: str,
        model: str | None,
        operation: str,
        rate_limit_group: str | None = None,
        deadline_monotonic: float | None = None,
    ) -> ProviderCallResult[ResultT]:
        """
        Execute a logical provider call and retain every physical attempt.

        Examples:
            Execute a successful single attempt:
                ```python
                result = await executor.execute(
                    call,
                    provider="openai",
                    model="gpt-5-nano",
                    operation="chat",
                )
                assert result.attempts[-1].outcome == "success"
                ```

            Coordinate an explicit provider rate-limit group:
                ```python
                result = await executor.execute(
                    call,
                    provider="azure",
                    model="deployment-a",
                    operation="embedding",
                    rate_limit_group="shared-deployment",
                )
                ```

        Args:
            call: Single-attempt provider adapter returning value and metadata.
            provider: Configured provider identifier.
            model: Provider model or deployment identifier when known.
            operation: Logical provider operation such as chat or embedding.
            rate_limit_group: Optional explicit key for shared provider quotas.
            deadline_monotonic: Optional caller deadline that retry waits may
                not cross.

        Returns:
            ProviderCallResult[ResultT]: Successful value, final provider
            metadata, and immutable physical-attempt history.

        Notes:
            HTTP 429 and pre-connect failures are retried by default. Read
            timeouts and 5xx responses are terminal because replay safety is
            unknown. Cancellation is never caught or translated.
        """

        start = self._clock()
        attempts: list[ProviderTransportAttempt] = []
        key = _rate_gate_key(
            provider,
            model,
            rate_limit_group,
            scope=self._rate_limit_scope,
        )
        settings = self.settings
        max_attempts = settings.max_attempts if settings.enabled else 1
        policy_deadline = start + settings.max_elapsed_s if settings.enabled else None
        effective_deadline = _earliest_deadline(policy_deadline, deadline_monotonic)

        for attempt_number in range(1, max_attempts + 1):
            try:
                await self.rate_gate.wait(
                    key,
                    deadline_monotonic=effective_deadline,
                )
            except ProviderRateGateDeadlineExceededError as exc:
                raise LLMProviderRequestError(
                    provider=provider,
                    model=model,
                    operation=operation,
                    code="provider_rate_gate_deadline_exceeded",
                    message=str(exc),
                    retryable=False,
                    attempts=tuple(attempts),
                ) from exc
            try:
                result = await call()
            except LLMProviderRequestError as error:
                request_error = error
            except httpx.TransportError as transport_error:
                request_error = classify_transport_error(
                    provider,
                    model,
                    operation,
                    transport_error,
                )
            else:
                attempts.append(
                    ProviderTransportAttempt(
                        attempt_number=attempt_number,
                        elapsed_s=max(0.0, self._clock() - start),
                        outcome="success",
                        retryable=False,
                        request_id=result.metadata.request_id,
                        rate_limits=result.metadata.rate_limits,
                    )
                )
                return ProviderCallResult(
                    value=result.value,
                    metadata=result.metadata,
                    attempts=tuple(attempts),
                )

            provider_delay_s = _provider_delay_s(request_error)
            scheduled_delay_s = self._scheduled_delay_s(
                attempt_number=attempt_number,
                provider_delay_s=provider_delay_s,
            )
            retry_allowed = (
                request_error.retryable
                and attempt_number < max_attempts
                and self._delay_fits_budget(
                    start=start,
                    delay_s=scheduled_delay_s,
                    provider_delay_s=provider_delay_s,
                    deadline_monotonic=effective_deadline,
                )
            )
            attempts.append(
                ProviderTransportAttempt(
                    attempt_number=attempt_number,
                    elapsed_s=max(0.0, self._clock() - start),
                    outcome="error",
                    retryable=retry_allowed,
                    status_code=request_error.status_code,
                    error_code=request_error.code,
                    request_id=request_error.metadata.request_id,
                    provider_delay_s=provider_delay_s,
                    scheduled_delay_s=scheduled_delay_s if retry_allowed else None,
                    rate_limits=request_error.metadata.rate_limits,
                )
            )
            if not retry_allowed:
                request_error.attempts = tuple(attempts)
                raise request_error
            await self.rate_gate.defer(key, scheduled_delay_s)

        raise AssertionError("provider retry loop terminated without a result")

    def _scheduled_delay_s(self, *, attempt_number: int, provider_delay_s: float) -> float:
        backoff_s = min(
            self.settings.max_backoff_s,
            self.settings.base_delay_s * (2 ** (attempt_number - 1)),
        )
        jitter_s = backoff_s * self.settings.jitter_ratio * _unit_interval(self._random_unit())
        return max(provider_delay_s, backoff_s + jitter_s)

    def _delay_fits_budget(
        self,
        *,
        start: float,
        delay_s: float,
        provider_delay_s: float,
        deadline_monotonic: float | None,
    ) -> bool:
        if provider_delay_s > self.settings.max_provider_delay_s:
            return False
        wake_at = self._clock() + delay_s
        if wake_at - start > self.settings.max_elapsed_s:
            return False
        return deadline_monotonic is None or wake_at <= deadline_monotonic


def _provider_delay_s(error: LLMProviderRequestError) -> float:
    candidates = [error.metadata.retry_after_s or 0.0]
    candidates.extend(snapshot.reset_after_s or 0.0 for snapshot in error.metadata.rate_limits)
    return max(candidates, default=0.0)


def _earliest_deadline(*deadlines: float | None) -> float | None:
    present = tuple(float(value) for value in deadlines if value is not None)
    return min(present) if present else None


def _rate_gate_key(
    provider: str,
    model: str | None,
    group: str | None,
    *,
    scope: str,
) -> str:
    return f"{provider.lower()}:{scope}:{group or model or 'default'}"


def _rate_limit_scope(base_url: str | None, credential: str | None) -> str:
    endpoint = _normalized_base_url(base_url)
    fingerprint = (
        hashlib.sha256(credential.encode("utf-8")).hexdigest()[:16] if credential else "anonymous"
    )
    return f"{endpoint}:{fingerprint}"


def _normalized_base_url(base_url: str | None) -> str:
    value = str(base_url or "default").strip().rstrip("/")
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def _unit_interval(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
