"""Shared provider-facing Tool discovery manifest rendering."""

from __future__ import annotations

from aethergraph.services.llm.tool_calling import ToolCallRequest


def render_tool_search_description(request: ToolCallRequest) -> str:
    """Render one compact provider-neutral Tool-search description.

    Intro:
        The description explains strict Engine search semantics and presents the
        authorized primary-path vocabulary consistently across providers.

    Examples:
        Render a request with deferred paths:
            ```python
            description = render_tool_search_description(request)
            assert "lexical_queries" in description
            ```

        Render a request without deferred paths:
            ```python
            description = render_tool_search_description(empty_request)
            assert description.endswith("none.")
            ```

    Args:
        request: Provider-neutral Tool-call request and deferred catalog.

    Returns:
        str: Compact search instructions and authorized primary paths.

    Notes:
        Exact deferred names live in the Engine stable capability index; this
        provider description does not create a second catalog authority.
    """

    paths = {
        tool.path.path: tool.path.description
        for tool in request.tools
        if tool.path is not None and tool.exposure == "deferred"
    }
    manifest = "; ".join(f"{path}: {description}" for path, description in sorted(paths.items()))
    prefix = (
        "Find authorized project Tools. Keep goal as semantic intent. For ranked "
        "search provide 1-4 concise keyword lexical_queries. Use complete_paths "
        "only for exhaustive authorized namespace selection. Select no more than "
        f"{request.discovery.max_results} Tools in one search; if a complete path "
        "contains more, choose a narrower path or use ranked search. Available primary "
        "capability paths"
    )
    return f"{prefix}: {manifest}." if manifest else f"{prefix}: none."


__all__ = ["render_tool_search_description"]
