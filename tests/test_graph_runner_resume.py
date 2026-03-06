import pytest

from aethergraph.contracts.errors.errors import GraphBuildError
from aethergraph.contracts.services.state_stores import GraphSnapshot
from aethergraph.core.graph.graph_spec import TaskGraphSpec
from aethergraph.core.graph.node_spec import TaskNodeSpec
from aethergraph.core.graph.node_state import NodeStatus
from aethergraph.core.runtime.graph_runner import (
    _prepare_resume_failed_nodes,
    _recover_graph_from_snapshot,
)


def _build_spec() -> TaskGraphSpec:
    return TaskGraphSpec(
        graph_id="resume-demo",
        nodes={
            "a": TaskNodeSpec(node_id="a", type="tool"),
            "b": TaskNodeSpec(node_id="b", type="tool", dependencies=["a"]),
            "c": TaskNodeSpec(node_id="c", type="tool", dependencies=["b"]),
        },
    )


def _build_snapshot() -> GraphSnapshot:
    return GraphSnapshot(
        run_id="run-parent",
        graph_id="resume-demo",
        rev=3,
        created_at=0.0,
        spec_hash="",
        state={
            "rev": 3,
            "_bound_inputs": {"x": 1},
            "nodes": {
                "a": {
                    "status": "DONE",
                    "outputs": {"result": 10},
                    "started_at": 1.0,
                    "finished_at": 2.0,
                },
                "b": {
                    "status": "FAILED",
                    "outputs": {"result": 20},
                    "error": "boom",
                    "attempts": 2,
                    "started_at": 3.0,
                    "finished_at": 4.0,
                },
                "c": {
                    "status": "SKIPPED",
                    "outputs": {},
                    "started_at": 5.0,
                    "finished_at": 6.0,
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_prepare_resume_failed_nodes_resets_failed_subgraph():
    spec = _build_spec()
    snap = _build_snapshot()
    snap.spec_hash = "ignored"
    graph = _recover_graph_from_snapshot(
        spec=spec,
        target_run_id="run-child",
        source_run_id="run-parent",
        snap=snap,
    )

    failed_roots = await _prepare_resume_failed_nodes(
        graph=graph,
        snap=snap,
        source_run_id="run-parent",
    )

    assert failed_roots == ["b"]
    assert graph.state.run_id == "run-child"
    assert graph.state.nodes["a"].status == NodeStatus.DONE
    assert graph.state.nodes["a"].outputs == {"result": 10}

    assert graph.state.nodes["b"].status == NodeStatus.PENDING
    assert graph.state.nodes["b"].outputs == {}
    assert graph.state.nodes["b"].error is None
    assert graph.state.nodes["b"].attempts == 0
    assert graph.state.nodes["b"].finished_at is None

    assert graph.state.nodes["c"].status == NodeStatus.PENDING
    assert graph.state.nodes["c"].outputs == {}
    assert graph.state.nodes["c"].finished_at is None


@pytest.mark.asyncio
async def test_prepare_resume_failed_nodes_requires_failed_roots():
    spec = _build_spec()
    snap = _build_snapshot()
    snap.state["nodes"]["b"]["status"] = "DONE"
    graph = _recover_graph_from_snapshot(
        spec=spec,
        target_run_id="run-child",
        source_run_id="run-parent",
        snap=snap,
    )

    with pytest.raises(GraphBuildError) as exc:
        await _prepare_resume_failed_nodes(
            graph=graph,
            snap=snap,
            source_run_id="run-parent",
        )

    assert exc.value.code == "resume_no_failed_nodes"
