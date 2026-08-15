import asyncio
import hashlib
import json
import os
import threading
import time

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent
from aethergraph.services.state_stores.scope import require_graph_run_scope
from aethergraph.storage.contracts.scope import StorageScope


class JsonGraphStateStore(GraphStateStore):
    def __init__(self, root: str):
        """Open the temporary scoped JSON graph-state store.

        Creates the configured root and initializes process-local synchronization.

        Examples:
            Open a test store:
                ```python
                store = JsonGraphStateStore(".data/graph-state")
                ```

            Inject a temporary store:
                ```python
                container.state_store = JsonGraphStateStore(workspace_path)
                ```

        Args:
            root: Authorized filesystem root for JSON graph state.

        Returns:
            None: The temporary store is ready.

        Notes:
            This implementation is removed at the S9 provider activation cut.
        """
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._alock = asyncio.Lock()
        self._tlock = threading.RLock()

    @staticmethod
    def _scope_token(scope: StorageScope) -> str:
        raw = json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _run_dir(self, scope: StorageScope, *, create: bool) -> str:
        d = os.path.join(self.root, f"v2-{self._scope_token(scope)}")
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    async def save_snapshot(self, scope: StorageScope, snap: GraphSnapshot) -> None:
        """Atomically save one scoped JSON graph snapshot.

        Writes a temporary file, flushes it, and replaces the final revision filename.

        Examples:
            Save a snapshot:
                ```python
                await store.save_snapshot(scope, snapshot)
                ```

            Save final state:
                ```python
                await store.save_snapshot(scope, final_snapshot)
                ```

        Args:
            scope: Exact canonical graph-run scope.
            snap: Complete graph snapshot matching the scope.

        Returns:
            None: The JSON snapshot is flushed before return.

        Notes:
            The scoped v2 directory has no fallback to the former run-only directory.
        """
        require_graph_run_scope(scope, run_id=snap.run_id, graph_id=snap.graph_id)
        d = self._run_dir(scope, create=True)
        ts = int(time.time())
        fn = f"snapshot_{snap.rev:08d}_{ts}.json"
        tmp = os.path.join(d, fn + ".tmp")
        dst = os.path.join(d, fn)
        with self._tlock:  # <— thread-safe region
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {**snap.__dict__, "_storage_scope": scope.as_filter()},
                    f,
                    ensure_ascii=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)

    async def load_latest_snapshot(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> GraphSnapshot | None:
        """Load the latest snapshot from one scoped JSON directory.

        Selects the lexically latest revision filename only after exact scope validation.

        Examples:
            Load a saved run:
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
            GraphSnapshot | None: Latest matching snapshot or `None`.

        Notes:
            Reads do not create missing directories.
        """
        require_graph_run_scope(scope, run_id=run_id)
        d = self._run_dir(scope, create=False)
        if not os.path.isdir(d):
            return None
        with self._tlock:
            files = [x for x in os.listdir(d) if x.startswith("snapshot_")]
            if not files:
                return None
            files.sort()
            with open(os.path.join(d, files[-1]), encoding="utf-8") as f:
                payload = json.load(f)
                payload.pop("_storage_scope", None)
                snapshot = GraphSnapshot(**payload)
                require_graph_run_scope(
                    scope,
                    run_id=snapshot.run_id,
                    graph_id=snapshot.graph_id,
                )
                return snapshot

    async def append_event(self, scope: StorageScope, ev: StateEvent) -> None:
        """Append one event to a scoped JSON-lines stream.

        Validates identity, writes one complete JSON line, and synchronizes the file.

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
            None: The event line is flushed before return.

        Notes:
            No unscoped JSON-lines file is written.
        """
        require_graph_run_scope(scope, run_id=ev.run_id, graph_id=ev.graph_id)
        p = os.path.join(self._run_dir(scope, create=True), "events.jsonl")
        line = (
            json.dumps({**ev.__dict__, "_storage_scope": scope.as_filter()}, ensure_ascii=False)
            + "\n"
        )
        with self._tlock, open(p, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    async def load_events_since(
        self,
        scope: StorageScope,
        run_id: str,
        from_rev: int,
    ) -> list[StateEvent]:
        """Load scoped JSON events after one graph revision.

        Reads one scoped JSON-lines file and retains events above the exclusive bound.

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
            list[StateEvent]: Matching events in file order.

        Notes:
            Missing event files return an empty list without probing legacy directories.
        """
        require_graph_run_scope(scope, run_id=run_id)
        p = os.path.join(self._run_dir(scope, create=False), "events.jsonl")
        if not os.path.exists(p):
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["rev"] > from_rev:
                    rec.pop("_storage_scope", None)
                    out.append(StateEvent(**rec))
        return out

    async def list_run_ids(
        self,
        scope: StorageScope,
        graph_id: str | None = None,
    ) -> list[str]:
        """List run identities from matching scoped JSON snapshot metadata.

        Temporarily scans only v2 directories and filters complete supplied scope and
        optional graph identity before returning sorted run IDs.

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
            scope: Canonical dimensions constraining visible JSON snapshots.
            graph_id: Optional exact graph identity filter.

        Returns:
            list[str]: Sorted matching run identities.

        Notes:
            The canonical implementation replaces this temporary filesystem scan.
        """
        runs: set[str] = set()
        expected = scope.as_filter()
        with self._tlock:
            for entry in os.listdir(self.root):
                directory = os.path.join(self.root, entry)
                if not entry.startswith("v2-") or not os.path.isdir(directory):
                    continue
                files = sorted(
                    name for name in os.listdir(directory) if name.startswith("snapshot_")
                )
                if not files:
                    continue
                with open(os.path.join(directory, files[-1]), encoding="utf-8") as handle:
                    payload = json.load(handle)
                stored_scope = dict(payload.get("_storage_scope") or {})
                if any(stored_scope.get(key) != value for key, value in expected.items()):
                    continue
                if graph_id is not None and payload.get("graph_id") != graph_id:
                    continue
                run_id = payload.get("run_id")
                if isinstance(run_id, str) and run_id:
                    runs.add(run_id)
        return sorted(runs)
