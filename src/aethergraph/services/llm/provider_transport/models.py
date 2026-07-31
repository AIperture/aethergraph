"""Provider-neutral transport result, error, and rate-limit contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from ..types import LLMError

ProviderAttemptOutcome = Literal["success", "error"]
ProviderRateLimitResource = Literal[
    "requests",
    "tokens",
    "input_tokens",
    "output_tokens",
    "unknown",
]


@dataclass(frozen=True)
class ProviderRateLimitSnapshot:
    """One provider-advertised rate-limit resource snapshot."""

    resource: ProviderRateLimitResource
    limit: int | None = None
    remaining: int | None = None
    reset_after_s: float | None = None


@dataclass(frozen=True)
class ProviderResponseMetadata:
    """Sanitized provider response metadata retained by AetherGraph."""

    request_id: str | None = None
    retry_after_s: float | None = None
    rate_limits: tuple[ProviderRateLimitSnapshot, ...] = ()


@dataclass(frozen=True)
class ProviderTransportAttempt:
    """One physical provider attempt within a logical LLM call."""

    attempt_number: int
    elapsed_s: float
    outcome: ProviderAttemptOutcome
    retryable: bool
    status_code: int | None = None
    error_code: str | None = None
    request_id: str | None = None
    provider_delay_s: float | None = None
    scheduled_delay_s: float | None = None
    rate_limits: tuple[ProviderRateLimitSnapshot, ...] = ()


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ProviderCallResult(Generic[ResultT]):
    """Successful provider value paired with normalized transport metadata."""

    value: ResultT
    metadata: ProviderResponseMetadata = ProviderResponseMetadata()


class LLMProviderRequestError(LLMError):
    """Canonical sanitized failure raised for a provider transport request."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        operation: str,
        code: str,
        message: str,
        retryable: bool,
        status_code: int | None = None,
        provider_error_code: str | None = None,
        provider_error_type: str | None = None,
        metadata: ProviderResponseMetadata | None = None,
        attempts: tuple[ProviderTransportAttempt, ...] = (),
    ) -> None:
        """
        Build one provider-neutral request failure.

        Examples:
            Represent a temporary provider throttle:
                ```python
                error = LLMProviderRequestError(
                    provider="openai",
                    model="gpt-5-nano",
                    operation="chat",
                    code="provider_rate_limited",
                    message="Rate limit reached.",
                    retryable=True,
                    status_code=429,
                )
                assert error.retryable is True
                ```

            Represent a permanent request rejection:
                ```python
                error = LLMProviderRequestError(
                    provider="local",
                    model=None,
                    operation="embedding",
                    code="provider_request_rejected",
                    message="Invalid request.",
                    retryable=False,
                    status_code=400,
                )
                assert error.status_code == 400
                ```

        Args:
            provider: Configured provider identifier.
            model: Provider model or deployment identifier when known.
            operation: Logical provider operation such as chat or embedding.
            code: Stable AetherGraph failure code.
            message: Sanitized human-readable provider failure summary.
            retryable: Whether the central retry policy may retry the failure.
            status_code: HTTP response status when a response was received.
            provider_error_code: Sanitized provider-native error code.
            provider_error_type: Sanitized provider-native error type.
            metadata: Normalized response metadata and advertised limits.
            attempts: Physical attempts already made for the logical call.

        Returns:
            None: Initializes the exception and its stable fields.

        Notes:
            Raw response bodies and headers are deliberately not retained.
            Retry execution may attach an immutable attempt history before the
            final exception leaves the AetherGraph LLM service.
        """

        super().__init__(message)
        self.provider = str(provider)
        self.model = str(model) if model is not None else None
        self.operation = str(operation)
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.provider_error_code = (
            str(provider_error_code) if provider_error_code is not None else None
        )
        self.provider_error_type = (
            str(provider_error_type) if provider_error_type is not None else None
        )
        self.metadata = metadata or ProviderResponseMetadata()
        self.attempts = tuple(attempts)
