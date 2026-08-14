from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.inspection import (
    RuntimeInspectionService,
    build_error_info,
)


def test_build_error_info_preserves_structured_traceback_fields():
    assert build_error_info(
        {
            "message": "Node exploded",
            "detail": "Traceback: node exploded",
            "kind": "runtime",
            "stage": "node_execution",
            "code": "ValueError",
            "hints": [{"code": "retry", "message": "Retry."}],
            "is_traceback": True,
        }
    ) == {
        "message": "Node exploded",
        "detail": "Traceback: node exploded",
        "kind": "runtime",
        "stage": "node_execution",
        "code": "ValueError",
        "hints": [{"code": "retry", "message": "Retry."}],
        "is_traceback": True,
    }


@pytest.mark.asyncio
async def test_runtime_inspection_merges_latest_snapshot_and_incremental_state():
    record = SimpleNamespace(
        run_id="run-1",
        graph_id="graph-1",
        status="failed",
        error="Run failed",
        meta={
            "error_info": {
                "message": "Run failed",
                "detail": "Traceback: run failed",
                "code": "RuntimeError",
                "is_traceback": True,
            }
        },
    )

    class _RunManager:
        async def get_record(self, run_id: str):
            return record if run_id == "run-1" else None

    class _StateStore:
        async def load_latest_snapshot(self, run_id: str):
            return SimpleNamespace(
                rev=3,
                state={
                    "nodes": {
                        "node-1": {
                            "status": "RUNNING",
                            "tool_name": "explode",
                            "started_at": "2026-07-18T00:00:00+00:00",
                        }
                    },
                    "edges": [{"from": "node-0", "to": "node-1"}],
                },
            )

        async def load_events_since(self, run_id: str, from_rev: int):
            assert from_rev == 3
            return [
                SimpleNamespace(
                    rev=4,
                    ts=1_752_796_801.0,
                    kind="STATUS",
                    payload={"node_id": "node-1", "status": "FAILED"},
                ),
                SimpleNamespace(
                    rev=5,
                    ts=1_752_796_802.0,
                    kind="OUTPUT",
                    payload={"node_id": "node-1", "outputs": None},
                ),
            ]

    state_store = _StateStore()
    inspection = await RuntimeInspectionService(
        run_manager=_RunManager(),
        state_store=state_store,
    ).inspect("run-1")

    assert inspection is not None
    assert inspection.record is record
    assert inspection.nodes_state["node-1"]["status"] == "FAILED"
    assert inspection.nodes_state["node-1"]["finished_at"] is not None
    assert inspection.snapshot_edges == ({"source": "node-0", "target": "node-1"},)
    assert inspection.run_error_info["detail"] == "Traceback: run failed"


@pytest.mark.asyncio
async def test_runtime_inspection_returns_node_error_diagnostics():
    record = SimpleNamespace(
        run_id="run-1",
        graph_id="graph-1",
        status="failed",
        error=None,
        meta={},
    )

    class _RunManager:
        async def get_record(self, run_id: str):
            return record

    class _StateStore:
        async def load_latest_snapshot(self, run_id: str):
            return SimpleNamespace(
                rev=1,
                state={
                    "nodes": {
                        "node-1": {
                            "status": "FAILED",
                            "tool_name": "explode",
                            "error": "Node exploded",
                            "error_info": {
                                "message": "Node exploded",
                                "detail": "Traceback: node exploded",
                                "code": "ValueError",
                                "is_traceback": True,
                            },
                        }
                    }
                },
            )

        async def load_events_since(self, run_id: str, from_rev: int):
            return []

    inspection = await RuntimeInspectionService(
        run_manager=_RunManager(),
        state_store=_StateStore(),
    ).inspect("run-1")

    assert inspection is not None
    diagnostic = inspection.node_diagnostics[0]
    assert diagnostic.node_id == "node-1"
    assert diagnostic.tool_name == "explode"
    assert diagnostic.error_info["detail"] == "Traceback: node exploded"
