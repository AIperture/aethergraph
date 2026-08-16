from __future__ import annotations

from aethergraph.contracts.services.state_stores import GraphSnapshot, StateEvent
from aethergraph.services.state_stores.json_store import JsonGraphStateStore
from aethergraph.storage.contracts.scope import StorageScope


async def test_json_graph_state_store_uses_the_exact_scope_contract(tmp_path) -> None:
    store = JsonGraphStateStore(str(tmp_path))
    scope = StorageScope(org_id="org-1", run_id="run-1", graph_id="graph-1")
    other_scope = StorageScope(org_id="org-2", run_id="run-1", graph_id="graph-1")
    snapshot = GraphSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        rev=1,
        created_at=0.0,
        spec_hash="abc",
        state={"nodes": {}},
    )
    event = StateEvent(
        run_id="run-1",
        graph_id="graph-1",
        rev=2,
        ts=123.0,
        kind="STATUS",
        payload={"node_id": "node-a", "status": "DONE"},
    )

    await store.save_snapshot(scope, snapshot)
    await store.append_event(scope, event)

    assert await store.load_latest_snapshot(scope, "run-1") == snapshot
    assert await store.load_latest_snapshot(other_scope, "run-1") is None
    assert await store.load_events_since(scope, "run-1", 1) == [event]
    assert await store.list_run_ids(StorageScope(org_id="org-1"), "graph-1") == ["run-1"]
