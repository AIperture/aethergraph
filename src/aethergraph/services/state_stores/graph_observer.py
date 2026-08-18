# aethergraph/persist/observer.py
from __future__ import annotations

import time
from typing import Any

from aethergraph.contracts.services.state_stores import GraphStateStore, StateEvent
from aethergraph.core.graph.task_node import NodeStatus
from aethergraph.services.state_stores.scope import require_graph_run_scope
from aethergraph.services.state_stores.serialize import _jsonish_outputs_with_refs
from aethergraph.services.state_stores.utils import snapshot_from_graph
from aethergraph.storage.contracts.scope import StorageScope


class PersistenceObserver:
    def __init__(
        self,
        *,
        store: GraphStateStore,
        scope: StorageScope,
        artifact_store: Any | None,
        spec_hash: str,
        snapshot_every: int = 50,
        min_interval_s: float = 5.0,
    ):
        """Attach canonical scope and persistence policy to graph callbacks.

        Retains one exact graph-run scope for every event and snapshot emitted by the
        observer and performs no provider selection.

        Examples:
            Persist every event interval:
                ```python
                observer = PersistenceObserver(
                    store=store,
                    scope=scope,
                    artifact_store=artifacts,
                    spec_hash=spec_hash,
                    snapshot_every=1,
                )
                ```

            Use the default snapshot cadence:
                ```python
                observer = PersistenceObserver(
                    store=store,
                    scope=scope,
                    artifact_store=artifacts,
                    spec_hash=spec_hash,
                )
                ```

        Args:
            store: Graph-state service receiving normalized persistence calls.
            scope: Exact canonical owner and graph-run identity.
            artifact_store: Artifact service used while serializing graph snapshots.
            spec_hash: Stable digest of the active graph specification.
            snapshot_every: Positive event interval between snapshot attempts.
            min_interval_s: Minimum elapsed seconds between snapshot writes.

        Returns:
            None: The observer is ready for graph registration.

        Notes:
            The observer never infers scope from `app_id` or writes a fallback store.
        """
        self.store = store
        self.scope = scope
        self.artifact_store = artifact_store
        self.spec_hash = spec_hash
        self.snapshot_every = snapshot_every
        self.min_interval_s = min_interval_s
        self._event_count = 0
        self._last_snap_ts = 0.0

    async def on_node_status_change(self, runtime_node):
        g = runtime_node._parent_graph
        ev = StateEvent(
            run_id=g.state.run_id or "unknown",
            graph_id=g.graph_id,
            rev=g.state.rev,
            ts=time.time(),
            kind="STATUS",
            payload={
                "node_id": runtime_node.node_id,
                "status": runtime_node.state.status.name
                if isinstance(runtime_node.state.status, NodeStatus)
                else str(runtime_node.state.status),
            },
        )
        require_graph_run_scope(self.scope, run_id=ev.run_id, graph_id=ev.graph_id)
        await self.store.append_event(self.scope, ev)
        await self._maybe_snapshot(g)

    async def on_node_output_change(self, runtime_node):
        g = runtime_node._parent_graph
        # make outputs JSON-safe for events (no externalization)
        safe_outputs = await _jsonish_outputs_with_refs(
            outputs=getattr(runtime_node.state, "outputs", None),
            run_id=g.state.run_id or "unknown",
            graph_id=g.graph_id,
            node_id=runtime_node.node_id,
            tool_name=getattr(runtime_node.state, "tool_name", None)
            or getattr(getattr(runtime_node, "spec", None), "tool_name", None),
            tool_version=getattr(runtime_node.state, "tool_version", None)
            or getattr(getattr(runtime_node, "spec", None), "tool_version", None),
            artifacts=None,  # ← keep events self-contained
            allow_externalize=False,  # ← do not write artifacts from events
        )

        ev = StateEvent(
            run_id=g.state.run_id or "unknown",
            graph_id=g.graph_id,
            rev=g.state.rev,
            ts=time.time(),
            kind="OUTPUT",
            payload={
                "node_id": runtime_node.node_id,
                "outputs": safe_outputs or {},  # ✅ JSON-safe
            },
        )
        require_graph_run_scope(self.scope, run_id=ev.run_id, graph_id=ev.graph_id)
        await self.store.append_event(self.scope, ev)
        await self._maybe_snapshot(g)

    async def on_inputs_bound(self, graph):
        # also sanitize inputs for events (in case user passed non-JSON)
        safe_inputs = await _jsonish_outputs_with_refs(
            outputs=getattr(graph.state, "_bound_inputs", None),
            run_id=graph.state.run_id or "unknown",
            graph_id=graph.graph_id,
            node_id="__graph_inputs__",
            tool_name=None,
            tool_version=None,
            artifacts=None,
            allow_externalize=False,
        )
        ev = StateEvent(
            run_id=graph.state.run_id or "unknown",
            graph_id=graph.graph_id,
            rev=graph.state.rev,
            ts=time.time(),
            kind="INPUTS_BOUND",
            payload={"inputs": safe_inputs or {}},  # JSON-safe
        )
        require_graph_run_scope(self.scope, run_id=ev.run_id, graph_id=ev.graph_id)
        await self.store.append_event(self.scope, ev)
        await self._maybe_snapshot(graph)

    async def on_patch_applied(self, graph, patch):
        ev = StateEvent(
            run_id=graph.state.run_id or "unknown",
            graph_id=graph.graph_id,
            rev=graph.state.rev,
            ts=time.time(),
            kind="PATCH",
            payload={"patch": patch.__dict__},
        )
        require_graph_run_scope(self.scope, run_id=ev.run_id, graph_id=ev.graph_id)
        await self.store.append_event(self.scope, ev)
        await self._maybe_snapshot(graph)

    async def _maybe_snapshot(self, graph):
        self._event_count += 1
        now = time.time()
        if (self._event_count % self.snapshot_every == 0) and (
            now - self._last_snap_ts >= self.min_interval_s
        ):
            snap = await snapshot_from_graph(
                run_id=graph.state.run_id or "unknown",
                graph_id=graph.graph_id,
                rev=graph.state.rev,
                spec_hash=self.spec_hash,
                state_obj=graph.state,
                graph_obj=graph,
                artifacts=self.artifact_store,
                allow_externalize=False,  # keep snapshots JSON-only (opaque refs)
                include_wait_spec=True,
            )
            require_graph_run_scope(self.scope, run_id=snap.run_id, graph_id=snap.graph_id)
            await self.store.save_snapshot(self.scope, snap)
            self._last_snap_ts = now
