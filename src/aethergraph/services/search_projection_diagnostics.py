"""Bounded diagnostics for authoritative search-projection failures."""

from __future__ import annotations

from aethergraph.services.llm.provider_transport import LLMProviderRequestError
from aethergraph.storage.contracts import StorageCapabilityError

_MAX_MESSAGE_CHARS = 500


def search_projection_diagnostic(exc: Exception) -> str:
    """Describe one search-projection failure without exposing arbitrary exception text.

    Known provider and capability failures retain sanitized canonical facts, while
    unknown exception messages remain hidden.

    Examples:
        Format a provider failure:
            ```python
            diagnostic = search_projection_diagnostic(provider_error)
            assert "provider=openai" in diagnostic
            ```

        Hide an unknown exception message:
            ```python
            diagnostic = search_projection_diagnostic(RuntimeError("private"))
            assert diagnostic == "RuntimeError: search projection failed"
            ```

    Args:
        exc: Exception raised while projecting an authoritative record into search.

    Returns:
        str: Bounded safe diagnostic suitable for durable projection intent state.

    Notes:
        `LLMProviderRequestError.message` is already sanitized by the provider
        transport contract; arbitrary exception messages are never copied.
    """

    if isinstance(exc, LLMProviderRequestError):
        facts = [
            "search projection failed",
            f"operation={exc.operation}",
            f"provider={exc.provider}",
            f"model={exc.model or 'unknown'}",
            f"code={exc.code}",
            f"retryable={str(exc.retryable).lower()}",
        ]
        if exc.status_code is not None:
            facts.append(f"status_code={exc.status_code}")
        if exc.provider_error_code is not None:
            facts.append(f"provider_error_code={exc.provider_error_code}")
        if exc.provider_error_type is not None:
            facts.append(f"provider_error_type={exc.provider_error_type}")
        message = " ".join(exc.message.split())[:_MAX_MESSAGE_CHARS]
        if message:
            facts.append(f"message={message}")
        return f"{type(exc).__name__}: " + "; ".join(facts)
    if isinstance(exc, StorageCapabilityError):
        missing = ",".join(exc.missing)
        return (
            f"{type(exc).__name__}: search projection failed; "
            f"provider={exc.provider_name}; missing={missing}"
        )
    return f"{type(exc).__name__}: search projection failed"


__all__ = ["search_projection_diagnostic"]
