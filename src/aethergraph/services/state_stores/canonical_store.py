"""Graph-state service over the canonical provider-owned storage primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import math
from typing import Any

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent
from aethergraph.services.state_stores.scope import require_graph_run_scope
from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    EventStore,
    PageRequest,
    RunQuery,
    RunRepository,
    SortDirection,
    StateStore,
    StorageCapacityError,
    StorageConflictError,
    StorageScope,
    StorageScopeError,
)

_GRAPH_STATE_NAMESPACE = "graph_state"
_GRAPH_SNAPSHOT_KEY = "latest"
_GRAPH_EVENT_KIND = "graph_state"
_PAGE_SIZE = 1_000
_MAX_QUERY_RECORDS = 10_000


class CanonicalGraphStateStore(GraphStateStore):
    """Graph snapshot and event facade using one canonical storage bundle."""

    def __init__(
        self,
        *,
        state_store: StateStore,
        event_store: EventStore,
        run_repository: RunRepository,
    ) -> None:
        """Compose graph state from canonical provider-owned repositories.

        Intro:
            Retains typed repository dependencies without opening storage, selecting a
            provider, or constructing a parallel persistence path.

        Examples:
            Build from one bundle:
            ```python
            store = CanonicalGraphStateStore(
                state_store=bundle.state,
                event_store=bundle.events,
                run_repository=bundle.runs,
            )
            ```

            Retain the service for runtime injection:
            ```python
            container.state_store = CanonicalGraphStateStore(
                state_store=state, event_store=events, run_repository=runs
            )
            ```

        Args:
            state_store: Canonical transactional current-state repository.
            event_store: Canonical ordered runtime event store.
            run_repository: Canonical indexed run repository.

        Returns:
            None: The facade is ready without performing I/O.

        Notes:
            Provider lifecycle remains owned by `StorageComposition`; this facade never
            closes individual stores.
        """
        self._state = state_store
        self._events = event_store
        self._runs = run_repository

    async def save_snapshot(self, scope: StorageScope, snap: GraphSnapshot) -> None:
        """Persist one latest graph snapshot with provider revision CAS.

        Intro:
            Writes the complete JSON snapshot into the dedicated graph-state namespace
            and rejects stale graph revisions before provider CAS.

        Examples:
            Save a first snapshot:
            ```python
            await store.save_snapshot(scope, snapshot)
            ```

            Save final graph outputs:
            ```python
            await store.save_snapshot(scope, final_snapshot)
            ```

        Args:
            scope: Exact canonical owner and run scope.
            snap: Complete graph snapshot matching the supplied scope.

        Returns:
            None: The current state row and provider history are durable.

        Notes:
            A graph revision is domain state; provider state revisions remain separate.
        """
        require_graph_run_scope(scope, run_id=snap.run_id, graph_id=snap.graph_id)
        value = _snapshot_value(snap)
        current = await self._state.get(scope, _GRAPH_STATE_NAMESPACE, _GRAPH_SNAPSHOT_KEY)
        expected_revision = current.revision if current is not None else 0
        if current is not None:
            current_snapshot = _snapshot_from_value(current.value)
            if current_snapshot.rev > snap.rev:
                raise StorageConflictError(
                    f"Graph snapshot revision cannot move backward from "
                    f"{current_snapshot.rev} to {snap.rev}"
                )
            if _thaw_json(current.value) == value:
                return
        await self._state.compare_and_set(
            scope,
            _GRAPH_STATE_NAMESPACE,
            _GRAPH_SNAPSHOT_KEY,
            expected_revision,
            value,
            {
                "graph_id": snap.graph_id,
                "graph_revision": snap.rev,
                "schema": "graph_snapshot.v1",
            },
        )

    async def load_latest_snapshot(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> GraphSnapshot | None:
        """Load one latest graph snapshot by exact canonical identity.

        Intro:
            Reads a single indexed current-state row and reconstructs the graph-domain
            snapshot without consulting events, documents, or legacy paths.

        Examples:
            Load an interrupted run:
            ```python
            snapshot = await store.load_latest_snapshot(scope, "run-1")
            ```

            Detect an empty run:
            ```python
            assert await store.load_latest_snapshot(scope, "run-1") is None
            ```

        Args:
            scope: Exact canonical owner and run scope.
            run_id: Exact stable run identity matching the scope.

        Returns:
            GraphSnapshot | None: Current graph snapshot or `None`.

        Notes:
            The lookup performs no broad event query or compatibility fallback.
        """
        require_graph_run_scope(scope, run_id=run_id)
        record = await self._state.get(scope, _GRAPH_STATE_NAMESPACE, _GRAPH_SNAPSHOT_KEY)
        if record is None:
            return None
        snapshot = _snapshot_from_value(record.value)
        require_graph_run_scope(scope, run_id=snapshot.run_id, graph_id=snapshot.graph_id)
        return snapshot

    async def append_event(self, scope: StorageScope, ev: StateEvent) -> None:
        """Append one normalized graph-state event to the runtime event stream.

        Intro:
            Promotes the graph event family and event kind into indexed fields while
            preserving authored graph revision and payload in canonical JSON.

        Examples:
            Append a node transition:
            ```python
            await store.append_event(scope, status_event)
            ```

            Append a graph patch:
            ```python
            await store.append_event(scope, patch_event)
            ```

        Args:
            scope: Exact canonical owner and run scope.
            ev: Graph event matching the supplied scope.

        Returns:
            None: The provider-assigned ordered event is durable.

        Notes:
            Deterministic content identity makes exact retries idempotent without dual writes.
        """
        require_graph_run_scope(scope, run_id=ev.run_id, graph_id=ev.graph_id)
        occurred_at = _event_time(ev.ts)
        payload = {
            "run_id": ev.run_id,
            "graph_id": ev.graph_id,
            "graph_revision": ev.rev,
            "event_kind": ev.kind,
            "data": ev.payload,
        }
        event_id = _event_id(scope=scope, occurred_at=occurred_at, payload=payload)
        await self._events.append(
            EventDraft(
                event_id=event_id,
                occurred_at=occurred_at,
                scope=scope,
                kind=_GRAPH_EVENT_KIND,
                stage=ev.kind.lower(),
                payload=payload,
            )
        )

    async def load_events_since(
        self,
        scope: StorageScope,
        run_id: str,
        from_rev: int,
    ) -> list[StateEvent]:
        """Load normalized graph events after one authored graph revision.

        Intro:
            Pages the exact run-scoped graph event family in ascending provider order,
            then applies the exclusive graph revision bound.

        Examples:
            Load events after a snapshot:
            ```python
            events = await store.load_events_since(scope, "run-1", snapshot.rev)
            ```

            Rebuild from the beginning:
            ```python
            events = await store.load_events_since(scope, "run-1", -1)
            ```

        Args:
            scope: Exact canonical owner and run scope.
            run_id: Exact stable run identity matching the scope.
            from_rev: Exclusive authored graph revision lower bound.

        Returns:
            list[StateEvent]: Matching events in ascending provider order.

        Notes:
            Reads fail above the explicit safety bound instead of truncating or scanning forever.
        """
        require_graph_run_scope(scope, run_id=run_id)
        records = []
        cursor: str | None = None
        while True:
            page = await self._events.query(
                EventQuery(
                    scope=scope,
                    page=PageRequest(limit=_PAGE_SIZE, cursor=cursor),
                    kinds=(_GRAPH_EVENT_KIND,),
                    order=SortDirection.ASCENDING,
                )
            )
            records.extend(page.items)
            if page.next_cursor is None:
                break
            if len(records) >= _MAX_QUERY_RECORDS:
                raise StorageCapacityError(
                    f"Graph event read exceeds {_MAX_QUERY_RECORDS} canonical records"
                )
            cursor = page.next_cursor

        events: list[StateEvent] = []
        for record in records:
            payload = _thaw_json(record.payload)
            graph_revision = payload.get("graph_revision")
            if not isinstance(graph_revision, int) or isinstance(graph_revision, bool):
                raise ValueError("Canonical graph event lacks an integer graph revision")
            if graph_revision <= from_rev:
                continue
            event = StateEvent(
                run_id=str(payload.get("run_id") or ""),
                graph_id=str(payload.get("graph_id") or ""),
                rev=graph_revision,
                ts=record.occurred_at.timestamp(),
                kind=str(payload.get("event_kind") or ""),
                payload=dict(payload.get("data") or {}),
            )
            require_graph_run_scope(
                scope,
                run_id=event.run_id,
                graph_id=event.graph_id,
            )
            events.append(event)
        return events

    async def list_run_ids(
        self,
        scope: StorageScope,
        graph_id: str | None = None,
    ) -> list[str]:
        """List owner-visible run identities through the indexed run repository.

        Intro:
            Applies canonical scope and optional graph filtering before cursor paging,
            replacing legacy snapshot-document enumeration.

        Examples:
            List owner runs:
            ```python
            run_ids = await store.list_run_ids(owner_scope)
            ```

            List one graph:
            ```python
            run_ids = await store.list_run_ids(owner_scope, graph_id="graph-1")
            ```

        Args:
            scope: Canonical scope constraining visible run records.
            graph_id: Optional exact graph identity filter.

        Returns:
            list[str]: Stable run identities in repository query order.

        Notes:
            Reads fail above the explicit safety bound and never inspect state keys.
        """
        if graph_id is not None and scope.graph_id is not None and scope.graph_id != graph_id:
            raise StorageScopeError(
                f"Graph list scope mismatch: scope={scope.graph_id!r}, value={graph_id!r}"
            )
        query_scope = replace(scope, graph_id=graph_id or scope.graph_id)
        run_ids: list[str] = []
        cursor: str | None = None
        while True:
            page = await self._runs.query(
                RunQuery(
                    scope=query_scope,
                    page=PageRequest(limit=_PAGE_SIZE, cursor=cursor),
                )
            )
            run_ids.extend(record.run_id for record in page.items)
            if page.next_cursor is None:
                return run_ids
            if len(run_ids) >= _MAX_QUERY_RECORDS:
                raise StorageCapacityError(
                    f"Graph run list exceeds {_MAX_QUERY_RECORDS} canonical records"
                )
            cursor = page.next_cursor


def _snapshot_value(snapshot: GraphSnapshot) -> dict[str, Any]:
    return {
        "run_id": snapshot.run_id,
        "graph_id": snapshot.graph_id,
        "graph_revision": snapshot.rev,
        "created_at": snapshot.created_at,
        "spec_hash": snapshot.spec_hash,
        "state": snapshot.state,
        "started_at": _encode_datetime(snapshot.started_at),
        "finished_at": _encode_datetime(snapshot.finished_at),
        "schema": "graph_snapshot.v1",
    }


def _snapshot_from_value(value: object) -> GraphSnapshot:
    payload = _thaw_json(value)
    if not isinstance(payload, dict):
        raise TypeError("Canonical graph snapshot value must be a mapping")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise TypeError("Canonical graph snapshot state must be a mapping")
    return GraphSnapshot(
        run_id=str(payload.get("run_id") or ""),
        graph_id=str(payload.get("graph_id") or ""),
        rev=int(payload["graph_revision"]),
        created_at=float(payload["created_at"]),
        spec_hash=str(payload.get("spec_hash") or ""),
        state=state,
        started_at=_decode_datetime(payload.get("started_at")),
        finished_at=_decode_datetime(payload.get("finished_at")),
    )


def _event_time(timestamp: float) -> datetime:
    if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
        raise TypeError("Graph event timestamp must be numeric")
    if not math.isfinite(float(timestamp)):
        raise ValueError("Graph event timestamp must be finite")
    return datetime.fromtimestamp(float(timestamp), tz=UTC)


def _event_id(
    *,
    scope: StorageScope,
    occurred_at: datetime,
    payload: Mapping[str, object],
) -> str:
    raw = json.dumps(
        {
            "scope": scope.as_filter(),
            "occurred_at": occurred_at.isoformat(),
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"graph-state-{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _encode_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _decode_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Canonical graph snapshot datetime must be an ISO string")
    return datetime.fromisoformat(value)


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
