from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aethergraph import NodeContext, graph_fn, graphify, tool
from aethergraph.core.runtime.graph_runner import run_async
from aethergraph.core.tools.waitable import DualStageTool


@tool(outputs=["graph_id"])
async def tool_with_ctx(*, ctx: NodeContext) -> dict[str, str]:
    return {"graph_id": ctx.graph_id}


@tool(outputs=["graph_id"])
async def tool_with_runtime(*, runtime: NodeContext) -> dict[str, str]:
    return {"graph_id": runtime.graph_id}


@pytest.mark.asyncio
async def test_graph_fn_injects_legacy_context_name():
    @graph_fn(name="test_ctx_injection_context_graph", inputs=[], outputs=["graph_id"])
    async def injected(*, context: NodeContext) -> dict[str, str]:
        return {"graph_id": context.graph_id}

    result = await run_async(injected, {})
    assert result["graph_id"] == "test_ctx_injection_context_graph"


@pytest.mark.asyncio
async def test_graph_fn_injects_ctx_alias():
    @graph_fn(name="test_ctx_injection_ctx_graph", inputs=[], outputs=["graph_id"])
    async def injected(*, ctx: NodeContext) -> dict[str, str]:
        return {"graph_id": ctx.graph_id}

    result = await run_async(injected, {})
    assert result["graph_id"] == "test_ctx_injection_ctx_graph"


@pytest.mark.asyncio
async def test_graph_fn_injects_arbitrary_typed_name():
    @graph_fn(name="test_ctx_injection_runtime_graph", inputs=[], outputs=["graph_id"])
    async def injected(*, runtime: NodeContext | None) -> dict[str, str]:
        assert runtime is not None
        return {"graph_id": runtime.graph_id}

    result = await run_async(injected, {})
    assert result["graph_id"] == "test_ctx_injection_runtime_graph"


@pytest.mark.asyncio
async def test_tool_node_injects_ctx_alias():
    @graphify(name="test_tool_ctx_alias_graph", inputs=[], outputs=["graph_id"])
    def tool_graph():
        out = tool_with_ctx()
        return {"graph_id": out.graph_id}

    result = await asyncio.wait_for(run_async(tool_graph, {}), timeout=5)
    assert result["graph_id"] == "test_tool_ctx_alias_graph"


@pytest.mark.asyncio
async def test_tool_node_injects_arbitrary_typed_name():
    @graphify(name="test_tool_runtime_alias_graph", inputs=[], outputs=["graph_id"])
    def tool_graph():
        out = tool_with_runtime()
        return {"graph_id": out.graph_id}

    result = await asyncio.wait_for(run_async(tool_graph, {}), timeout=5)
    assert result["graph_id"] == "test_tool_runtime_alias_graph"


def test_waitable_inference_excludes_typed_node_context_inputs():
    @tool(outputs=["decision"])
    class ApprovalWait(DualStageTool):
        outputs = ["decision"]

        async def setup(self, prompt: str, *, ctx: NodeContext) -> dict[str, Any]:
            return {
                "kind": "approval",
                "prompt": {"text": prompt},
                "resume_schema": {"type": "object"},
            }

        async def on_resume(
            self, resume: dict[str, Any], *, runtime: NodeContext
        ) -> dict[str, str]:
            return {"decision": "approved"}

    assert ApprovalWait.__aether_inputs__ == ["prompt"]


@pytest.mark.asyncio
async def test_nested_graph_call_preserves_explicit_context_alias():
    @graph_fn(name="test_nested_explicit_context_inner", inputs=[], outputs=["graph_id"])
    async def inner(*, runtime: NodeContext) -> dict[str, str]:
        return {"graph_id": runtime.graph_id}

    @graph_fn(name="test_nested_explicit_context_outer", inputs=[], outputs=["graph_id"])
    async def outer(*, context: NodeContext) -> dict[str, str]:
        return await inner(context=context)

    result = await run_async(outer, {})
    assert result["graph_id"] == "test_nested_explicit_context_outer"


@pytest.mark.asyncio
async def test_nested_graph_call_preserves_explicit_ctx_alias():
    @graph_fn(name="test_nested_explicit_ctx_inner", inputs=[], outputs=["graph_id"])
    async def inner(*, runtime: NodeContext) -> dict[str, str]:
        return {"graph_id": runtime.graph_id}

    @graph_fn(name="test_nested_explicit_ctx_outer", inputs=[], outputs=["graph_id"])
    async def outer(*, context: NodeContext) -> dict[str, str]:
        return await inner(ctx=context)

    result = await run_async(outer, {})
    assert result["graph_id"] == "test_nested_explicit_ctx_outer"


def test_ambiguous_node_context_signature_fails_fast():
    with pytest.raises(TypeError, match="multiple NodeContext parameters"):

        @graph_fn(name="test_ambiguous_ctx_graph", inputs=[], outputs=["value"])
        async def ambiguous(*, context: NodeContext, runtime: NodeContext) -> dict[str, str]:
            return {"value": "nope"}


@pytest.mark.asyncio
async def test_untyped_arbitrary_name_remains_regular_input():
    @graph_fn(name="test_untyped_runtime_input_graph", inputs=["runtime"], outputs=["value"])
    async def passthrough(runtime: str) -> dict[str, str]:
        return {"value": runtime}

    result = await run_async(passthrough, {"runtime": "user-value"})
    assert result["value"] == "user-value"
