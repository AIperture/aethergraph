# aethergraph/persist/interfaces.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from aethergraph.storage.contracts.scope import StorageScope


@dataclass
class GraphSnapshot:
    run_id: str
    graph_id: str
    rev: int
    created_at: float  # epoch seconds
    spec_hash: str  # detect spec drift
    state: dict[str, Any]  # JSON-serializable TaskGraphState
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class StateEvent:
    run_id: str
    graph_id: str
    rev: int
    ts: float
    kind: str  # "STATUS" | "OUTPUT" | "INPUTS_BOUND" | "PATCH"
    payload: dict[str, Any]


class GraphStateStore(Protocol):
    async def save_snapshot(self, scope: StorageScope, snap: GraphSnapshot) -> None:
        """Persist one latest graph snapshot within exact canonical run scope.

        Intro:
            The store validates that `scope.run_id` matches the snapshot and keeps owner
            dimensions attached to the authoritative state identity.

        Examples:
            Save a current snapshot:
                ```python
                await store.save_snapshot(scope, snapshot)
                ```

            Save a later graph revision:
                ```python
                await store.save_snapshot(scope, next_snapshot)
                ```

        Args:
            scope: Canonical owner and run scope matching the snapshot identity.
            snap: Complete JSON-compatible graph snapshot.

        Returns:
            None: The latest canonical snapshot is durable before return.

        Notes:
            Implementations use provider CAS and never infer owner scope from `run_id`.
        """
        ...

    async def load_latest_snapshot(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> GraphSnapshot | None:
        """Load the latest snapshot for one exact canonical run scope.

        Intro:
            Reads remain constrained to the supplied owner dimensions and never search a
            different tenant, project, organization, or user for the same run identity.

        Examples:
            Load an interrupted run:
                ```python
                snapshot = await store.load_latest_snapshot(scope, "run-1")
                ```

            Detect no saved snapshot:
                ```python
                assert await store.load_latest_snapshot(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and run scope matching `run_id`.
            run_id: Exact stable run identity.

        Returns:
            GraphSnapshot | None: Latest snapshot or `None` within that exact scope.

        Notes:
            A numeric graph revision is not a provider state revision or lookup alias.
        """
        ...

    async def append_event(self, scope: StorageScope, ev: StateEvent) -> None:
        """Append one graph-state event within exact canonical run scope.

        Intro:
            Event order is provider-owned while the authored graph revision remains in the
            event payload for incremental reconstruction.

        Examples:
            Append a status transition:
                ```python
                await store.append_event(scope, status_event)
                ```

            Append a graph patch:
                ```python
                await store.append_event(scope, patch_event)
                ```

        Args:
            scope: Canonical owner and run scope matching the event identity.
            ev: Complete graph-state event to append.

        Returns:
            None: The event is durable before return.

        Notes:
            Implementations do not write events to a fallback log or unscoped stream.
        """
        ...

    async def load_events_since(
        self,
        scope: StorageScope,
        run_id: str,
        from_rev: int,
    ) -> list[StateEvent]:
        """Load graph-state events authored after one graph revision.

        Intro:
            The exact canonical scope filters before ordering and revision selection so a
            same-named run in another owner scope is never visible.

        Examples:
            Load events after a snapshot:
                ```python
                events = await store.load_events_since(scope, "run-1", snapshot.rev)
                ```

            Load all authored events:
                ```python
                events = await store.load_events_since(scope, "run-1", -1)
                ```

        Args:
            scope: Canonical owner and run scope matching `run_id`.
            run_id: Exact stable run identity.
            from_rev: Exclusive authored graph revision lower bound.

        Returns:
            list[StateEvent]: Matching events in stable ascending authored order.

        Notes:
            Provider cursors establish storage order; `from_rev` remains graph-domain state.
        """
        ...

    async def list_run_ids(
        self,
        scope: StorageScope,
        graph_id: str | None = None,
    ) -> list[str]:
        """List bounded owner-visible run identities with optional graph filtering.

        Intro:
            Canonical implementations query the provider run repository instead of
            scanning snapshot document identifiers.

        Examples:
            List runs for one owner:
                ```python
                run_ids = await store.list_run_ids(owner_scope)
                ```

            List one graph's runs:
                ```python
                run_ids = await store.list_run_ids(owner_scope, graph_id="graph-1")
                ```

        Args:
            scope: Canonical owner scope constraining visible runs.
            graph_id: Optional exact graph identity filter.

        Returns:
            list[str]: Stable owner-visible run identities.

        Notes:
            Unscoped global enumeration and document-ID scans are not canonical behavior.
        """
        ...
