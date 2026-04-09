from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aethergraph import NodeContext, graph_fn, tool
from aethergraph.core.runtime.graph_runner import run_async
from aethergraph.core.tools.waitable import DualStageTool


@tool(outputs=["value"])
def runtime_sync_tool(x: int) -> dict[str, int]:
    return {"value": x + 1}


@tool(outputs=["value"])
async def runtime_async_tool(x: int) -> dict[str, int]:
    await asyncio.sleep(0)
    return {"value": x * 2}


@tool(outputs=["decision"])
class RuntimeApprovalWait(DualStageTool):
    outputs = ["decision"]

    async def setup(self, prompt: str, *, context: NodeContext) -> dict[str, Any]:
        return {
            "kind": "approval",
            "prompt": {"text": prompt},
            "resume_schema": {"type": "object"},
        }

    async def on_resume(self, resume: dict[str, Any], *, context: NodeContext) -> dict[str, str]:
        return {"decision": "approved"}


@pytest.mark.asyncio
async def test_graph_fn_runs_sync_and_async_tools_in_immediate_mode():
    @graph_fn(name="graph_fn.runtime.tool_mix", inputs=["x"], outputs=["result"])
    async def mixed_tools(x: int, *, context: NodeContext) -> dict[str, int]:
        sync_out = runtime_sync_tool(x=x)
        async_out = await runtime_async_tool(x=x)
        return {"result": sync_out["value"] + async_out["value"] + len(context.graph_id)}

    result = await run_async(mixed_tools, {"x": 3})
    assert result == {"result": 4 + 6 + len("graph_fn.runtime.tool_mix")}


@pytest.mark.asyncio
async def test_async_tool_immediate_mode_returns_raw_coroutine():
    coro = runtime_async_tool(x=5)
    assert asyncio.iscoroutine(coro)
    assert await coro == {"value": 10}


@pytest.mark.asyncio
async def test_immediate_mode_rejects_graph_control_kwargs():
    @graph_fn(name="graph_fn.runtime.control_kwargs", inputs=["x"], outputs=["result"])
    async def invalid_control(x: int) -> dict[str, int]:
        return runtime_sync_tool(x=x, _after="not-allowed")

    with pytest.raises(RuntimeError, match="tool_control_kwargs_build_mode_only"):
        await run_async(invalid_control, {"x": 1})


@pytest.mark.asyncio
async def test_immediate_mode_rejects_waitable_tools():
    @graph_fn(name="graph_fn.runtime.waitable", inputs=["prompt"], outputs=["decision"])
    async def invalid_waitable(prompt: str) -> dict[str, str]:
        return await RuntimeApprovalWait(prompt=prompt)

    with pytest.raises(RuntimeError, match="waitable_tool_immediate_mode_unsupported"):
        await run_async(invalid_waitable, {"prompt": "Proceed?"})


@pytest.mark.asyncio
async def test_graph_fn_rejects_node_handle_like_outputs():
    class FakeHandle:
        node_id = "fake"
        output_keys = ["value"]

    @graph_fn(name="graph_fn.runtime.fake_handle", inputs=[], outputs=["value"])
    async def invalid_output() -> FakeHandle:
        return FakeHandle()

    with pytest.raises(ValueError, match="graph_fn_plain_runtime_only"):
        await run_async(invalid_output, {})


@pytest.mark.asyncio
async def test_graph_fn_rejects_ref_like_outputs():
    @graph_fn(name="graph_fn.runtime.fake_ref", inputs=[], outputs=["value"])
    async def invalid_output() -> dict[str, Any]:
        return {"value": {"_type": "ref", "from": "node_1", "key": "value"}}

    with pytest.raises(ValueError, match="graph_fn_plain_runtime_only"):
        await run_async(invalid_output, {})


@pytest.mark.asyncio
async def test_graph_fn_single_output_allows_plain_value():
    @graph_fn(name="graph_fn.runtime.single_value", inputs=["x"], outputs=["result"])
    async def single_value(x: int) -> int:
        return x * 3

    result = await run_async(single_value, {"x": 7})
    assert result == {"result": 21}


@pytest.mark.asyncio
async def test_graph_fn_multiple_outputs_require_dict():
    @graph_fn(name="graph_fn.runtime.multi_value", inputs=["x"], outputs=["a", "b"])
    async def invalid_shape(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="graph_fn_result_shape_invalid"):
        await run_async(invalid_shape, {"x": 2})


@pytest.mark.asyncio
async def test_run_async_graph_fn_executes_without_interpreter_runtime():
    @graph_fn(name="graph_fn.runtime.basic", inputs=["value"], outputs=["result"])
    async def basic(value: str) -> dict[str, str]:
        return {"result": value.upper()}

    result = await run_async(basic, {"value": "hello"})
    assert result == {"result": "HELLO"}
