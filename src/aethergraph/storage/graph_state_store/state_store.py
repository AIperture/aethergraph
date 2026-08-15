import hashlib
import json
import time

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.state_stores.scope import require_graph_run_scope
from aethergraph.storage.contracts.scope import StorageScope


class GraphStateStoreImpl(GraphStateStore):
    """
    Generic GraphStateStore implementation that combines a DocStore for snapshots
    - DocStore for storing GraphSnapshot documents
    - EventLog for storing StateEvent logs
    """

    def __init__(self, *, doc_store: "DocStore", event_log: "EventLog"):
        """Compose the temporary legacy graph-state implementation.

        Retains the injected document and event stores without selecting or opening
        another backend.

        Examples:
            Compose local dependencies:
                ```python
                store = GraphStateStoreImpl(doc_store=docs, event_log=events)
                ```

            Inject the service into a container:
                ```python
                container.state_store = GraphStateStoreImpl(doc_store=docs, event_log=events)
                ```

        Args:
            doc_store: Legacy document store used for current snapshots.
            event_log: Legacy append-only event log used for graph events.

        Returns:
            None: The composed service is ready without performing I/O.

        Notes:
            This implementation is deleted at the S9 provider activation cut.
        """
        self._docs = doc_store
        self._log = event_log

    @staticmethod
    def _scope_token(scope: StorageScope) -> str:
        raw = json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _snapshot_id(self, scope: StorageScope) -> str:
        return f"graph_state/v2/{self._scope_token(scope)}/latest"

    async def save_snapshot(self, scope: StorageScope, snap: GraphSnapshot) -> None:
        """Save a snapshot under the clean scoped v2 legacy key.

        Validates graph identity before replacing the one latest legacy document.

        Examples:
            Save a snapshot:
                ```python
                await store.save_snapshot(scope, snapshot)
                ```

            Save final outputs:
                ```python
                await store.save_snapshot(scope, final_snapshot)
                ```

        Args:
            scope: Exact canonical graph-run scope.
            snap: Complete graph snapshot matching the scope.

        Returns:
            None: The legacy snapshot document is durable before return.

        Notes:
            Old unscoped document keys are never probed or rewritten.
        """
        require_graph_run_scope(scope, run_id=snap.run_id, graph_id=snap.graph_id)
        await self._docs.put(
            self._snapshot_id(scope),
            {**snap.__dict__, "_storage_scope": scope.as_filter()},
        )

    async def load_latest_snapshot(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> GraphSnapshot | None:
        """Load the latest scoped v2 legacy snapshot.

        Reads only the deterministic key derived from the complete supplied scope.

        Examples:
            Load an existing snapshot:
                ```python
                snapshot = await store.load_latest_snapshot(scope, "run-1")
                ```

            Detect absence:
                ```python
                assert await store.load_latest_snapshot(scope, "missing") is None
                ```

        Args:
            scope: Exact canonical graph-run scope.
            run_id: Stable run identity matching the scope.

        Returns:
            GraphSnapshot | None: Matching snapshot or `None`.

        Notes:
            The former run-only key is intentionally not a fallback.
        """
        require_graph_run_scope(scope, run_id=run_id)
        # The saved snapshot is always the latest so just fetch by fixed id
        doc = await self._docs.get(self._snapshot_id(scope))
        if not doc:
            return None
        payload = dict(doc)
        payload.pop("_storage_scope", None)
        snapshot = GraphSnapshot(**payload)
        require_graph_run_scope(scope, run_id=snapshot.run_id, graph_id=snapshot.graph_id)
        return snapshot

    async def append_event(self, scope: StorageScope, ev: StateEvent) -> None:
        """Append one event under the scoped v2 legacy stream identity.

        Normalizes the legacy envelope while retaining the authored graph-event kind.

        Examples:
            Append status:
                ```python
                await store.append_event(scope, status_event)
                ```

            Append output:
                ```python
                await store.append_event(scope, output_event)
                ```

        Args:
            scope: Exact canonical graph-run scope.
            ev: Graph event matching the scope.

        Returns:
            None: The legacy event is durable before return.

        Notes:
            No event is duplicated to an unscoped stream.
        """
        require_graph_run_scope(scope, run_id=ev.run_id, graph_id=ev.graph_id)
        # standard event log append
        payload = ev.__dict__.copy()
        payload["scope_id"] = f"graph_state:v2:{self._scope_token(scope)}"
        payload["_storage_scope"] = scope.as_filter()
        payload["event_kind"] = payload.get("kind")
        payload["kind"] = "graph_state"
        payload.setdefault("ts", time.time())
        await self._log.append(payload)

    async def load_events_since(
        self,
        scope: StorageScope,
        run_id: str,
        from_rev: int,
    ) -> list[StateEvent]:
        """Load scoped legacy events after one graph revision.

        Queries only the v2 stream identity and filters by the authored revision.

        Examples:
            Load incremental events:
                ```python
                events = await store.load_events_since(scope, "run-1", snapshot.rev)
                ```

            Load all events:
                ```python
                events = await store.load_events_since(scope, "run-1", -1)
                ```

        Args:
            scope: Exact canonical graph-run scope.
            run_id: Stable run identity matching the scope.
            from_rev: Exclusive authored graph revision lower bound.

        Returns:
            list[StateEvent]: Matching events in legacy query order.

        Notes:
            Older event-kind envelopes are intentionally unsupported after the clean cut.
        """
        require_graph_run_scope(scope, run_id=run_id)
        rows = await self._log.query(
            scope_id=f"graph_state:v2:{self._scope_token(scope)}",
            kinds=["graph_state"],
            # from_rev filter will be applied below
        )
        out = []
        for row in rows:
            if row.get("rev", -1) > from_rev:
                out.append(
                    StateEvent(
                        run_id=row.get("run_id", run_id),
                        graph_id=row.get("graph_id", ""),
                        rev=row.get("rev", -1),
                        ts=row.get("ts", time.time()),
                        kind=row.get("event_kind") or row.get("kind") or "PATCH",
                        payload=row.get("payload") or {},
                    )
                )
        return out

    async def list_run_ids(
        self,
        scope: StorageScope,
        graph_id: str | None = None,
    ) -> list[str]:
        """List scoped v2 legacy snapshot run identities.

        Temporarily scans legacy document metadata while filtering every supplied scope
        dimension and optional graph identity.

        Examples:
            List owner runs:
                ```python
                run_ids = await store.list_run_ids(owner_scope)
                ```

            Filter by graph:
                ```python
                run_ids = await store.list_run_ids(owner_scope, graph_id="graph-1")
                ```

        Args:
            scope: Canonical dimensions constraining visible legacy snapshots.
            graph_id: Optional exact graph identity filter.

        Returns:
            list[str]: Sorted matching run identities.

        Notes:
            The canonical implementation replaces this scan with `RunRepository.query`.
        """
        ids = await self._docs.list()
        runs: set[str] = set()
        for doc_id in ids:
            if not doc_id.startswith("graph_state/v2/"):
                continue
            doc = await self._docs.get(doc_id)
            if not doc:
                continue
            stored_scope = dict(doc.get("_storage_scope") or {})
            expected = scope.as_filter()
            if any(stored_scope.get(key) != value for key, value in expected.items()):
                continue
            if graph_id is not None and doc.get("graph_id") != graph_id:
                continue
            run_id = doc.get("run_id")
            if isinstance(run_id, str) and run_id:
                runs.add(run_id)
        return sorted(runs)
