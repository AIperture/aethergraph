from __future__ import annotations

from aethergraph.services.llm.provider_transport import LLMProviderRequestError
from aethergraph.services.search_projection_diagnostics import (
    search_projection_diagnostic,
)


def test_provider_projection_diagnostic_preserves_sanitized_failure_facts() -> None:
    diagnostic = search_projection_diagnostic(
        LLMProviderRequestError(
            provider="openai",
            model="text-embedding-3-small",
            operation="embedding",
            code="provider_request_rejected",
            message="The configured project cannot access this model.",
            retryable=False,
            status_code=403,
            provider_error_code="model_not_found",
            provider_error_type="invalid_request_error",
        )
    )

    assert "operation=embedding" in diagnostic
    assert "provider=openai" in diagnostic
    assert "model=text-embedding-3-small" in diagnostic
    assert "code=provider_request_rejected" in diagnostic
    assert "status_code=403" in diagnostic
    assert "provider_error_code=model_not_found" in diagnostic
    assert "message=The configured project cannot access this model." in diagnostic


def test_unknown_projection_diagnostic_hides_arbitrary_exception_text() -> None:
    diagnostic = search_projection_diagnostic(
        RuntimeError("provider unavailable with secret-value")
    )

    assert diagnostic == "RuntimeError: search projection failed"
    assert "secret-value" not in diagnostic
