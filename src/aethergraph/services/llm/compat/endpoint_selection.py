"""Legacy endpoint-less Chat binding compatibility."""

from __future__ import annotations

from aethergraph.services.llm.registry import (
    EndpointAdapterDescriptor,
    resolve_endpoint_adapter,
)


def resolve_legacy_chat_adapter(
    provider: str,
    endpoint_id: str | None,
    *,
    has_tool_request: bool,
) -> EndpointAdapterDescriptor:
    """Resolve one exact adapter at the endpoint-less `0.1.x` boundary.

    Intro:
        Explicit bindings always win. Endpoint-less profiles use the registry
        default except legacy Azure Tool requests, which preserve their existing
        Responses route until Studio settings v4 requires explicit assignments.

    Examples:
        Resolve endpoint-less Azure direct Chat:
            ```python
            adapter = resolve_legacy_chat_adapter(
                "azure",
                None,
                has_tool_request=False,
            )
            assert adapter.adapter_id == "azure_chat_completions"
            ```

        Preserve endpoint-less Azure Tool routing:
            ```python
            adapter = resolve_legacy_chat_adapter(
                "azure",
                None,
                has_tool_request=True,
            )
            assert adapter.adapter_id == "azure_responses"
            ```

    Args:
        provider: Exact registered provider identity.
        endpoint_id: Optional explicitly pinned Chat adapter identity.
        has_tool_request: Whether the invocation carries a native Tool contract.

    Returns:
        EndpointAdapterDescriptor: Exact adapter selected for this invocation.

    Notes:
        This is a versioned compatibility decision, not provider fallback. It is
        the sole request-dependent endpoint selector and must be deleted when the
        endpoint-less Studio v3 boundary is retired.
    """

    if endpoint_id is None and provider == "azure" and has_tool_request:
        return resolve_endpoint_adapter(provider, "chat", endpoint_id="azure_responses")
    return resolve_endpoint_adapter(provider, "chat", endpoint_id=endpoint_id)


__all__ = ["resolve_legacy_chat_adapter"]
