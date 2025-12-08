# services/artifacts/facade.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from aethergraph.contracts.services.artifacts import Artifact, AsyncArtifactStore
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.artifacts.paths import _from_uri_or_path
from aethergraph.services.scope.scope import Scope

ArtifactView = Literal["node", "graph", "run", "all"]


class ArtifactFacade:
    """
    Facade for artifact storage + indexing within a specific execution context.

    - All *writes* go through the underlying AsyncArtifactStore AND AsyncArtifactIndex.
    - Adds scoping helpers for search/list/best.
    - Provides backend-agnostic "as_local_*" helpers that work with FS and S3.
    """

    def __init__(
        self,
        *,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        store: AsyncArtifactStore,
        index: AsyncArtifactIndex,
        scope: Scope | None = None,
    ) -> None:
        self.run_id = run_id
        self.graph_id = graph_id
        self.node_id = node_id
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.store = store
        self.index = index

        # set scope -- this should be done outside in NodeContext and passed in, but here is a fallback
        self.scope = scope
        print("🍎", self.scope)

        # Keep track of the last created artifact
        self.last_artifact: Artifact | None = None

    # ---------- Helpers for scopes ----------
    def _with_scope_labels(self, labels: dict[str, Any] | None) -> dict[str, Any]:
        """Merge given labels with scope labels."""
        out: dict[str, Any] = dict(labels or {})
        if self.scope:
            out.update(self.scope.artifact_scope_labels())
        print(out)
        return out

    def _tenant_labels_for_search(self) -> dict[str, Any]:
        """
        Tenant filter for search/list.
        In cloud/demo mode, we AND these on.
        In local mode, these are no-ops.
        """
        if self.scope is None:
            return {}

        if self.scope.mode == "local":
            return {}

        labels: dict[str, Any] = {}
        if self.scope.org_id:
            labels["org_id"] = self.scope.org_id
        if self.scope.user_id:
            labels["user_id"] = self.scope.user_id
        if self.scope.client_id:
            labels["client_id"] = self.scope.client_id
        return labels

    def _view_labels(self, view: ArtifactView) -> dict[str, Any]:
        """Labels to filter by for a given ArtifactView.
        view options:
          - "node": filter by (run_id, graph_id, node_id)
          - "graph": filter by (run_id, graph_id)
          - "run": filter by (run_id)   [default]
          - "all": no implicit filters

          In cloud/demo mode, we AND tenant filters on.
          In local mode, tenants are no-ops.
        """
        base: dict[str, Any] = {}

        if view == "node":
            base = {"run_id": self.run_id, "graph_id": self.graph_id, "node_id": self.node_id}
        elif view == "graph":
            base = {"run_id": self.run_id, "graph_id": self.graph_id}
        elif view == "run":
            base = {"run_id": self.run_id}
        # "all" => no run/graph/node filter

        base.update(self._tenant_labels_for_search())
        print("🍊", base)
        return base

    # Metering-enhanced record
    async def _record(self, a: Artifact) -> None:
        """Record artifact in index and occurrence log."""
        # 1) Sync cononical tenant fields from labels/scope into artifact
        if self.scope:
            scope_labels = self.scope.artifact_scope_labels()

            # Ensure labels contains scope info
            a.labels = {**scope_labels, **(a.labels or {})}

            # Fill canoncial fields if missing
            a.org_id = a.org_id or self.scope.get("org_id")
            a.user_id = a.user_id or self.scope.get("user_id")
            a.client_id = a.client_id or self.scope.get("client_id")
            a.app_id = a.app_id or self.scope.get("app_id")
            a.session_id = a.session_id or self.scope.get("session_id")
            # run_id/graph_id/node_id should already be set

        # 2) Record in index + occurrence log
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a

        # 3) Metering hook for artifact writes
        try:
            meter = current_metering()
            ctx = current_meter_context.get() or {}

            run_id = getattr(a, "run_id", self.run_id)
            graph_id = getattr(a, "graph_id", self.graph_id)
            user_id = None
            org_id = None

            if self.scope is not None:
                dims = self.scope.metering_dimensions()
                user_id = dims.get("user_id", ctx.get("user_id"))
                org_id = dims.get("org_id", ctx.get("org_id"))
                run_id = dims.get("run_id", run_id)
                graph_id = dims.get("graph_id", graph_id)
            else:
                user_id = ctx.get("user_id")
                org_id = ctx.get("org_id")

            size = (
                getattr(a, "bytes", None)
                or getattr(a, "size_bytes", None)
                or getattr(a, "size", None)
                or 0
            )

            await meter.record_artifact(
                user_id=user_id,
                org_id=org_id,
                run_id=run_id,
                graph_id=graph_id,
                kind=getattr(a, "kind", "unknown"),
                bytes=int(size),
                pinned=bool(getattr(a, "pinned", False)),
            )
        except Exception:
            import logging

            logging.getLogger("aethergraph.metering").exception("record_artifact_failed")

    # ---------- core staging/ingest ----------
    async def stage_path(self, ext: str = "") -> str:
        """Plan a staging path in the underlying store."""
        return await self.store.plan_staging_path(planned_ext=ext)

    async def stage_dir(self, suffix: str = "") -> str:
        """Plan a staging directory in the underlying store."""
        return await self.store.plan_staging_dir(suffix=suffix)

    async def ingest_file(
        self,
        staged_path: str,
        *,
        kind: str,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        pin: bool = False,
    ) -> Artifact:
        """Ingest a staged file and record it in the index."""
        labels = self._with_scope_labels(labels)
        a = await self.store.ingest_staged_file(
            staged_path=staged_path,
            kind=kind,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )
        await self._record(a)
        return a

    async def ingest_dir(
        self,
        staged_dir: str,
        **kwargs: Any,
    ) -> Artifact:
        """
        Turn a staged directory into a directory artifact with manifest (and optional archive),
        then index it.
        Additional kwargs are passed to store.ingest_directory (kind, labels, etc.).
        """
        labels = self._with_scope_labels(kwargs.pop("labels", None))
        kwargs["labels"] = labels
        a = await self.store.ingest_directory(
            staged_dir=staged_dir,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            **kwargs,
        )
        await self._record(a)
        return a

    # ---------- core save APIs ----------
    async def save_file(
        self,
        path: str,
        *,
        kind: str,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        pin: bool = False,
        cleanup: bool = True,
    ) -> Artifact:
        """Save an existing file and index it."""
        labels = self._with_scope_labels(labels)
        a = await self.store.save_file(
            path=path,
            kind=kind,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
            cleanup=cleanup,
        )
        await self._record(a)
        return a

    async def save_text(
        self,
        payload: str,
        *,
        suggested_uri: str | None = None,
        kind: str = "text",
        labels: dict | None = None,
        metrics: dict | None = None,
        pin: bool = False,
    ) -> Artifact:
        """
        Save a text payload as an artifact with full context metadata.

        Implementation:
          - stage a temp .txt file
          - write payload
          - call save_file(kind="text", ...)
        """
        staged = await self.stage_path(".txt")

        def _write() -> str:
            p = Path(staged)
            p.write_text(payload, encoding="utf-8")
            return str(p)

        staged = await asyncio.to_thread(_write)

        return await self.save_file(
            path=staged,
            kind=kind,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )

    async def save_json(
        self,
        payload: dict,
        *,
        suggested_uri: str | None = None,
        kind: str = "json",
        labels: dict | None = None,
        metrics: dict | None = None,
        pin: bool = False,
    ) -> Artifact:
        """
        Save a JSON payload as an artifact with full context metadata.

        Implementation:
          - stage a temp .json file
          - write JSON
          - call save_file(kind="json", ...)
        """
        staged = await self.stage_path(".json")

        def _write() -> str:
            p = Path(staged)
            # Use ensure_ascii=False to preserve unicode; tweak as you like
            import json

            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(p)

        staged = await asyncio.to_thread(_write)

        return await self.save_file(
            path=staged,
            kind=kind,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )

    # ---------- streaming APIs ----------
    @asynccontextmanager
    async def writer(
        self,
        *,
        kind: str,
        planned_ext: str | None = None,
        pin: bool = False,
    ) -> AsyncIterator[Any]:
        """
        Async contextmanager yielding a writer object with:
            writer.write(bytes)
            writer.add_labels(...)
            writer.add_metrics(...)

        After context exit, writer.artifact will be populated by the store,
        and we will record it in the index here.
        """
        # 1) Delegate to the store's async context manager
        async with self.store.open_writer(
            kind=kind,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            planned_ext=planned_ext,
            pin=pin,
        ) as w:
            # 2) Yield to user code (they write() and add_labels/add_metrics)
            yield w

        # 3) At this point, store.open_writer has fully exited and has set w.artifact
        a = getattr(w, "artifact", None) or getattr(w, "_artifact", None)

        if a:
            await self._record(a)
        else:
            self.last_artifact = None

    # ---------- load APIs ----------
    async def load_bytes(self, uri: str) -> bytes:
        """Load raw bytes for a file-like artifact URI."""
        return await self.store.load_bytes(uri)

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        return await self.store.load_text(uri, encoding=encoding, errors=errors)

    async def load_json(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        text = await self.load_text(uri, encoding=encoding, errors=errors)
        return json.loads(text)

    async def load_artifact(self, uri: str) -> Any:
        """Compatibility helper: returns bytes or directory path depending on implementation."""
        return await self.store.load_artifact(uri)

    async def load_artifact_bytes(self, uri: str) -> bytes:
        return await self.store.load_artifact_bytes(uri)

    async def load_artifact_dir(self, uri: str) -> str:
        """
        Backend-agnostic: ensure a directory artifact is available as a local dir path.

        FS backend can just return its CAS dir; S3 backend might download to a temp dir.
        """
        return await self.store.load_artifact_dir(uri)

    # ---------- as local helpers ----------
    async def as_local_dir(
        self,
        artifact_or_uri: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        Ensure an artifact representing a directory is available as a local path.

        - For FS, usually returns the underlying CAS directory.
        - For S3, implementation should download to staging and return that path.
        """
        uri = artifact_or_uri.uri if isinstance(artifact_or_uri, Artifact) else str(artifact_or_uri)
        path = await self.store.load_artifact_dir(uri)
        if must_exist and not Path(path).exists():
            raise FileNotFoundError(f"Local path for artifact dir not found: {path}")
        return str(Path(path).resolve())

    async def as_local_file(
        self,
        artifact_or_uri: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        Ensure an artifact representing a single file is available as a local file path.

        - If uri is file:// or local path → return directly.
        - Otherwise (e.g., s3://) → download bytes into a staging file and return that path.
        """
        uri = artifact_or_uri.uri if isinstance(artifact_or_uri, Artifact) else str(artifact_or_uri)
        u = urlparse(uri)

        # local fs
        if not u.scheme or u.scheme.lower() == "file":
            path = _from_uri_or_path(uri).resolve()
            if must_exist and not Path(path).exists():
                raise FileNotFoundError(f"Local path for artifact file not found: {path}")
            if must_exist and not Path(path).is_file():
                raise FileNotFoundError(f"Local path for artifact file is not a file: {path}")
            return path

        # Non-FS backend: download to staging
        data = await self.store.load_artifact_bytes(uri)
        staged = await self.store.plan_staging_path(".bin")

        def _write():
            p = Path(staged)
            p.write_bytes(data)
            return str(p.resolve())

        path = await asyncio.to_thread(_write)
        return path

    # ---------- indexing helpers ----------
    async def list(self, *, view: ArtifactView = "run") -> list[Artifact]:
        """
        Quick listing scoped to current run/graph/node by default.
        view options:
          - "node": filter by (run_id, graph_id, node_id)
          - "graph": filter by (run_id, graph_id)
          - "run": filter by (run_id)   [default]
          - "all": no implicit filters
        """
        if view == "all":
            # still tenant-scoped
            labels = self._tenant_labels_for_search()
            return await self.index.search(labels=labels or None)
        labels = self._view_labels(view)
        return await self.index.search(labels=labels or None)

    async def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, str] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
        view: ArtifactView = "run",
        extra_scope_labels: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        """Pass-thorough search with scoping."""
        eff_labels: dict[str, str] = dict(labels or {})
        eff_labels.update(self._view_labels(view))
        if extra_scope_labels:
            eff_labels.update(extra_scope_labels)

        return await self.index.search(
            kind=kind,
            labels=eff_labels or None,
            metric=metric,
            mode=mode,
            limit=limit,
        )

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        view: ArtifactView = "run",
        filters: dict[str, str] | None = None,
    ) -> Artifact | None:
        eff_filters: dict[str, str] = dict(filters or {})
        eff_filters.update(self._view_labels(view))

        return await self.index.best(
            kind=kind,
            metric=metric,
            mode=mode,
            filters=eff_filters or None,
        )

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        """Mark/unmark an artifact as pinned."""
        await self.index.pin(artifact_id, pinned=pinned)

    # ---------- internal helpers ----------
    async def _record_simple(self, a: Artifact) -> None:
        """Record artifact in index and occurrence log."""
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a

    def _scope_labels(self, scope: Scope) -> dict[str, Any]:
        if scope == "node":
            return {"run_id": self.run_id, "graph_id": self.graph_id, "node_id": self.node_id}
        if scope == "graph":
            return {"run_id": self.run_id, "graph_id": self.graph_id}
        if scope == "run":
            return {"run_id": self.run_id}
        return {}  # "all"

    # ---------- deprecated / compatibility ----------
    async def stage(self, ext: str = "") -> str:
        """DEPRECATED: use stage_path()."""
        return await self.stage_path(ext=ext)

    async def ingest(
        self,
        staged_path: str,
        *,
        kind: str,
        labels=None,
        metrics=None,
        suggested_uri: str | None = None,
        pin: bool = False,
    ):  # DEPRECATED: use ingest_file()
        return await self.ingest_file(
            staged_path,
            kind=kind,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )

    async def save(
        self,
        path: str,
        *,
        kind: str,
        labels=None,
        metrics=None,
        suggested_uri: str | None = None,
        pin: bool = False,
    ):  # DEPRECATED: use save_file()
        return await self.save_file(
            path,
            kind=kind,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )

    async def tmp_path(self, suffix: str = "") -> str:  # DEPRECATED: use stage_path()
        return await self.stage_path(suffix)

    # FS-only, legacy helpers — prefer as_local_dir/as_local_file for new code
    def to_local_path(
        self,
        uri_or_path: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        DEPRECATED (FS-only):

        This assumes file:// or plain local paths; will not work correctly with s3://.
        Use `await as_local_dir(...)` or `await as_local_file(...)` instead.
        """
        s = uri_or_path.uri if isinstance(uri_or_path, Artifact) else str(uri_or_path)
        p = _from_uri_or_path(s).resolve()

        u = urlparse(s)
        if "://" in s and (u.scheme or "").lower() != "file":
            # Non-FS backend – just return the URI string
            return s

        if must_exist and not p.exists():
            raise FileNotFoundError(f"Local path not found: {p}")
        return str(p)

    def to_local_file(
        self,
        uri_or_path: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """DEPRECATED: FS-only; use `await as_local_file(...)` instead."""
        p = Path(self.to_local_path(uri_or_path, must_exist=must_exist))
        if must_exist and not p.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {p}")
        return str(p)

    def to_local_dir(
        self,
        uri_or_path: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """DEPRECATED: FS-only; use `await as_local_dir(...)` instead."""
        p = Path(self.to_local_path(uri_or_path, must_exist=must_exist))
        if must_exist and not p.is_dir():
            raise NotADirectoryError(f"Expected directory, got file: {p}")
        return str(p)
