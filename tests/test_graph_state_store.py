from aethergraph.contracts.services.state_stores import GraphSnapshot, StateEvent
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

    snap = GraphSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        rev=1,
        created_at=0.0,
        spec_hash="abc",
        state={"nodes": {}},
    )
    await store.save_snapshot(snap)

    ev = StateEvent(
        run_id="run-1",
        graph_id="graph-1",
        rev=2,
        ts=123.0,
        kind="STATUS",
        payload={"node_id": "node-a", "status": "RUNNING"},
    )
    await store.append_event(ev)

    loaded = await store.load_events_since("run-1", 1)
    assert len(loaded) == 1
    assert loaded[0].kind == "STATUS"
    assert loaded[0].payload == {"node_id": "node-a", "status": "RUNNING"}

    # Ensure the event log envelope is queryable by the generic graph_state kind.
    assert event_log.rows[0]["kind"] == "graph_state"
    assert event_log.rows[0]["event_kind"] == "STATUS"
