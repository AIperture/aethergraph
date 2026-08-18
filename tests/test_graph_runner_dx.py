from __future__ import annotations

import asyncio

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.contracts.errors.errors import GraphBuildError, GraphInputBindError
from aethergraph.core.graph.graph_builder import graph as graph_ctx
from aethergraph.core.graph.graph_refs import arg
import aethergraph.core.runtime.graph_runner as graph_runner
from aethergraph.core.runtime.graph_runner import run_async
from aethergraph.services.container.default_container import build_default_container


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path):
    container = build_default_container(
        root=str(tmp_path),
        cfg=AppSettings(workspace=str(tmp_path)),
    )
    asyncio.run(container.start_storage())
    monkeypatch.setattr(graph_runner, "_get_container", lambda: container)
    yield
    asyncio.run(container.close_storage())


@pytest.mark.asyncio
async def test_run_async_invalid_target_is_build_error():
    with pytest.raises(GraphBuildError) as exc:
        await run_async(123, {})
    err = exc.value
    assert err.stage == "materialization"
    assert err.code == "run_async_invalid_target"
    assert isinstance(err.hints, list)
    assert err.hints


@pytest.mark.asyncio
async def test_run_async_missing_required_inputs_is_build_error():
    with graph_ctx(name="missing_inputs_graph") as g:
        g.declare_inputs(required=["x"], optional={})
        g.expose("x_out", arg("x"))

    with pytest.raises(GraphInputBindError) as exc:
        await run_async(g, {})
    err = exc.value
    assert err.stage == "input_bind"
    assert err.code == "graph_inputs_missing_required"
    assert any(h.get("code") == "provide_required_inputs" for h in err.hints)
