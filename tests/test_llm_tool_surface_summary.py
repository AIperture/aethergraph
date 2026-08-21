from __future__ import annotations

from aethergraph.services.llm.tool_calling import (
    TOOL_SURFACE_SUMMARY_VERSION,
    ModelToolSpec,
    ToolCallRequest,
    tool_call_surface_summary,
)
from aethergraph.services.llm.tool_discovery import ToolDiscoveryRequest


def _tool(name: str, *, exposure: str) -> ModelToolSpec:
    return ModelToolSpec(
        name=name,
        description=f"Run {name}.",
        input_schema={"type": "object", "properties": {}},
        exposure=exposure,
    )


def test_tool_surface_separates_immediate_activated_and_searchable_tools() -> None:
    request = ToolCallRequest(
        tools=(
            _tool("skill_search", exposure="immediate"),
            _tool("authoring_read_bundle", exposure="deferred"),
            _tool("build_compile", exposure="deferred"),
        ),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn-1",
        active_tool_names=("authoring_read_bundle",),
    )

    surface = tool_call_surface_summary(request)

    assert surface is not None
    assert surface["schema_version"] == TOOL_SURFACE_SUMMARY_VERSION
    assert surface["callable_count"] == 2
    assert surface["immediate_count"] == 1
    assert surface["activated_deferred_count"] == 1
    assert surface["searchable_count"] == 1
    assert surface["callable_count"] + surface["searchable_count"] == len(surface["tools"])
    assert [(tool["name"], tool["exposure"], tool["callable"]) for tool in surface["tools"]] == [
        ("skill_search", "immediate", True),
        ("authoring_read_bundle", "deferred", True),
        ("build_compile", "deferred", False),
    ]
    assert all("active" not in tool for tool in surface["tools"])
    assert "active_count" not in surface


def test_tool_surface_is_absent_without_native_tool_request() -> None:
    assert tool_call_surface_summary(None) is None
