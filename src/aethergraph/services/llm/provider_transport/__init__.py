"""Provider-neutral HTTP transport contracts and classification."""

from .classification import (
    classify_http_error,
    classify_transport_error,
    provider_response_metadata,
)
from .models import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateLimitSnapshot,
    ProviderResponseMetadata,
    ProviderRetrySettings,
    ProviderTransportAttempt,
)
from .rate_gate import ProviderRateGate
from .retry import ProviderRetryExecutor

__all__ = [
    "LLMProviderRequestError",
    "ProviderCallResult",
    "ProviderRateLimitSnapshot",
    "ProviderRateGate",
    "ProviderResponseMetadata",
    "ProviderRetryExecutor",
    "ProviderRetrySettings",
    "ProviderTransportAttempt",
    "classify_http_error",
    "classify_transport_error",
    "provider_response_metadata",
]
