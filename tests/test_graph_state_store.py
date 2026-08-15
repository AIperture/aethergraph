import pytest

from aethergraph.contracts.services.state_stores import GraphSnapshot, StateEvent
from aethergraph.services.state_stores.json_store import JsonGraphStateStore
from aethergraph.storage.contracts.errors import StorageScopeError
from aethergraph.storage.contracts.scope import StorageScope
from aethergraph.storage.graph_state_store.state_store import GraphStateStoreImpl


class FakeDocStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    async def put(self, doc_id: str, value: dict) -> None:
        self.docs[doc_id] = value

    async def get(self, doc_id: str):
        return self.docs.get(doc_id)

    async def list(self):
        return list(self.docs.keys())


class FakeEventLog:
    def __init__(self):
        self.rows: list[dict] = []

    async def append(self, payload: dict) -> None:
        self.rows.append(dict(payload))

    async def query(self, *, scope_id=None, kinds=None, **kwargs):
        out = list(self.rows)
        if scope_id is not None:
            out = [row for row in out if row.get("scope_id") == scope_id]
        if kinds:
            allowed = set(kinds)
            out = [row for row in out if row.get("kind") in allowed]
        return out


async def test_graph_state_store_round_trips_state_events():
    docs = FakeDocStore()
    event_log = FakeEventLog()
    store = GraphStateStoreImpl(doc_store=docs, event_log=event_log)
    scope = StorageScope(org_id="org-1", user_id="user-1", run_id="run-1", graph_id="graph-1")

    snap = GraphSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        rev=1,
        created_at=0.0,
        spec_hash="abc",
        state={"nodes": {}},
    )
    await store.save_snapshot(scope, snap)

    ev = StateEvent(
        run_id="run-1",
        graph_id="graph-1",
        rev=2,
        ts=123.0,
        kind="STATUS",
        payload={"node_id": "node-a", "status": "RUNNING"},
    )
    await store.append_event(scope, ev)

    loaded = await store.load_events_since(scope, "run-1", 1)
    assert len(loaded) == 1
    assert loaded[0].kind == "STATUS"
    assert loaded[0].payload == {"node_id": "node-a", "status": "RUNNING"}

    # Ensure the event log envelope is queryable by the generic graph_state kind.
    assert event_log.rows[0]["kind"] == "graph_state"
    assert event_log.rows[0]["event_kind"] == "STATUS"


@pytest.mark.asyncio
async def test_graph_state_store_isolates_same_run_id_by_owner_scope():
    docs = FakeDocStore()
    store = GraphStateStoreImpl(doc_store=docs, event_log=FakeEventLog())
    first_scope = StorageScope(org_id="org-1", run_id="run-1", graph_id="graph-1")
    second_scope = StorageScope(org_id="org-2", run_id="run-1", graph_id="graph-1")
    snapshot = GraphSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        rev=1,
        created_at=0.0,
        spec_hash="abc",
        state={"owner": "org-1"},
    )

    await store.save_snapshot(first_scope, snapshot)

    assert await store.load_latest_snapshot(first_scope, "run-1") == snapshot
    assert await store.load_latest_snapshot(second_scope, "run-1") is None


@pytest.mark.asyncio
async def test_graph_state_store_rejects_mismatched_run_scope():
    store = GraphStateStoreImpl(doc_store=FakeDocStore(), event_log=FakeEventLog())

    with pytest.raises(StorageScopeError, match="run scope mismatch"):
        await store.load_latest_snapshot(StorageScope(run_id="run-2"), "run-1")


@pytest.mark.asyncio
async def test_json_graph_state_store_uses_the_same_exact_scope_contract(tmp_path):
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
