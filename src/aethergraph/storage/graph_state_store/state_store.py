import time

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog


class GraphStateStoreImpl(GraphStateStore):
    """
    Generic GraphStateStore implementation that combines a DocStore for snapshots
    - DocStore for storing GraphSnapshot documents
    - EventLog for storing StateEvent logs
    """

    def __init__(self, *, doc_store: "DocStore", event_log: "EventLog"):
        self._docs = doc_store
        self._log = event_log

    def _snapshot_id(self, run_id: str) -> str:
        return f"graph_state/{run_id}/latest"

    async def save_snapshot(self, snap: GraphSnapshot) -> None:
        # TODO: consider add history of snapshots by rev/timestamp
        # e.g. hist_id = f"graph_state/{run_id}/rev_{snap.rev:08d}_{int(snap.created_at)}"
        # self._docs.put(hist_id, snap.__dict__)
        # but this is not needed for retrieval of latest snapshot
        await self._docs.put(self._snapshot_id(snap.run_id), snap.__dict__)

    async def load_latest_snapshot(self, run_id) -> GraphSnapshot | None:
        # The saved snapshot is always the latest so just fetch by fixed id
        doc = await self._docs.get(self._snapshot_id(run_id))
        return GraphSnapshot(**doc) if doc else None

    async def append_event(self, ev: StateEvent) -> None:
        # standard event log append
        payload = ev.__dict__.copy()
        payload.setdefault("scope_id", ev.run_id)
        payload.setdefault("kind", "graph_state")
        payload.setdefault("ts", time.time())
        await self._log.append(payload)

    async def load_events_since(self, run_id, from_rev) -> list[StateEvent]:
        rows = await self._log.query(
            scope_id=run_id,
            kinds=["graph_state"],
            # from_rev filter will be applied below
        )
        out = []
        for row in rows:
            if row.get("rev", -1) > from_rev:
                out.append(StateEvent(**row))
        return out

    async def list_run_ids(self, graph_id: str | None = None) -> list[str]:
        # Basic version: ask DocStore for all ids and filter. This is sufficient for local/file-based stores.
        # In cloud implementations, this should be optimized with proper indexing
        ids = await self._docs.list()
        runs: set[str] = set()
        for doc_id in ids:
            if not doc_id.startswith("graph_state/"):
                continue
            _, run_id, *_ = doc_id.split("/")
            runs.add(run_id)
        return sorted(runs)
