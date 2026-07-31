"""Provider-neutral HTTP transport contracts and classification."""

from .classification import classify_http_error, provider_response_metadata
from .models import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateLimitSnapshot,
    ProviderResponseMetadata,
    ProviderTransportAttempt,
)

__all__ = [
    "LLMProviderRequestError",
    "ProviderCallResult",
    "ProviderRateLimitSnapshot",
    "ProviderResponseMetadata",
    "ProviderTransportAttempt",
    "classify_http_error",
    "provider_response_metadata",
]
