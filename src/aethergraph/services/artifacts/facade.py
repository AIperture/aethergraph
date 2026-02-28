# services/artifacts/facade.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
import warnings

from aethergraph.contracts.services.artifacts import Artifact, AsyncArtifactStore
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.search_backend import ScoredItem, SearchMode
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.artifacts.paths import _from_uri_or_path
from aethergraph.services.artifacts.types import ArtifactContent, ArtifactSearchResult
from aethergraph.services.artifacts.utils import _infer_content_mode
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.scope.scope import Scope, ScopeLevel
from aethergraph.storage.vector_index.utils import build_index_meta_from_scope


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
        art_store: AsyncArtifactStore,
        art_index: AsyncArtifactIndex,
        scoped_indices: ScopedIndices | None = None,
        scope: Scope | None = None,
    ) -> None:
        """
        Initialize an artifact facade bound to the current execution context.

        The facade binds run/graph/node identity and service handles so all
        artifact writes and lookups can be scoped consistently.

        Examples:
            Constructing the facade inside runtime wiring:
            ```python
            facade = ArtifactFacade(
                run_id=run_id,
                graph_id=graph_id,
                node_id=node_id,
                tool_name="my_tool",
                tool_version="0.1.0",
                art_store=store,
                art_index=index,
            )
            ```

        Args:
            run_id: Current run identifier.
            graph_id: Current graph identifier.
            node_id: Current node identifier.
            tool_name: Logical tool name associated with artifact writes.
            tool_version: Tool version associated with artifact writes.
            art_store: Artifact store implementation for blob persistence.
            art_index: Artifact index implementation for metadata search/list.
            scoped_indices: Optional semantic/lexical search backend wrapper.
            scope: Optional scope object carrying tenant/run/session labels.

        Returns:
            None: Initializes instance state.
        """
        self.run_id = run_id
        self.graph_id = graph_id
        self.node_id = node_id
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.store = art_store
        self.index = art_index
        self.scoped_indices = scoped_indices

        # set scope -- this should be done outside in NodeContext and passed in, but here is a fallback
        self.scope = scope

        # Keep track of the last created artifact
        self.last_artifact: Artifact | None = None

    # ---------- Helpers for scopes ----------
    def _with_scope_labels(self, labels: dict[str, Any] | None) -> dict[str, Any]:
        """
        Merge caller labels with scope-derived artifact labels.

        Scope labels are appended so downstream indexing/search can apply tenant
        and scope boundaries consistently.

        Examples:
            Merge user labels with runtime scope labels:
            ```python
            merged = facade._with_scope_labels({"kind": "report"})
            ```

        Args:
            labels: Optional user-provided labels dictionary.

        Returns:
            dict[str, Any]: Effective labels including scope labels when available.
        """
        out: dict[str, Any] = dict(labels or {})
        if self.scope:
            out.update(self.scope.artifact_scope_labels())
        return out

    def _tenant_labels_for_search(self) -> dict[str, Any]:
        """
        Build tenant labels used by search/list filters.

        In non-local mode this includes available org/user/client identifiers.
        In local mode this returns an empty dict.

        Examples:
            Derive tenant labels for index filtering:
            ```python
            tenant = facade._tenant_labels_for_search()
            ```

        Args:
            None.

        Returns:
            dict[str, Any]: Tenant labels to AND into structured filters.
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

    def _filters_for_level(self, level: ScopeLevel) -> dict[str, Any]:
        """
        Build scope filters for a given scope level.

        This normalizes scope-derived labels for `scope/session/run/user/org`
        so query methods can apply consistent boundaries.

        Examples:
            Build run-level filters:
            ```python
            where = facade._filters_for_level("run")
            ```

        Args:
            level: Requested scope level.

        Returns:
            dict[str, Any]: Normalized filter dictionary with non-null values only.
        """
        if self.scope is None:
            return {}

        base = self.scope.rag_filter(scope_id=self.scope.memory_scope_id())

        if not level or level == "scope":
            return {k: v for k, v in base.items() if v is not None}

        if level == "session" and self.scope.session_id:
            base["session_id"] = self.scope.session_id
        elif level == "run" and self.scope.run_id:
            base["run_id"] = self.scope.run_id
        elif level == "user":
            u = self.scope.user_id or self.scope.client_id
            if u:
                base["user_id"] = u
        elif level == "org" and self.scope.org_id:
            base["org_id"] = self.scope.org_id

        return {k: v for k, v in base.items() if v is not None}

    # Metering-enhanced record
    async def _record(self, a: Artifact) -> None:
        """
        Record a persisted artifact across indexing, search, and metering layers.

        This method normalizes timestamps/labels, writes index records, syncs
        scoped search indices, records metering, and updates run/session stores.

        Examples:
            Record an artifact returned by store.save_file:
            ```python
            artifact = await facade.store.save_file(...)
            await facade._record(artifact)
            ```

        Args:
            a: Artifact object to record.

        Returns:
            None: Persists side effects in index, metering, and runtime stores.
        """
        # 1) Sync canonical tenant fields from labels/scope into artifact
        if self.scope is not None:
            scope_labels = self.scope.artifact_scope_labels()
            a.labels = {**scope_labels, **(a.labels or {})}

            dims = self.scope.metering_dimensions()
            a.org_id = a.org_id or dims.get("org_id")
            a.user_id = a.user_id or dims.get("user_id")
            a.client_id = a.client_id or dims.get("client_id")
            a.app_id = a.app_id or dims.get("app_id")
            a.session_id = a.session_id or dims.get("session_id")
            # run_id / graph_id / node_id are already set

        # 1.1) Normalize timestamp: ensure a.created_at is a UTC datetime
        created_dt: datetime
        if isinstance(a.created_at, datetime):
            if a.created_at.tzinfo is None:
                created_dt = a.created_at.replace(tzinfo=timezone.utc)
            else:
                created_dt = a.created_at.astimezone(timezone.utc)
        elif isinstance(a.created_at, str):
            # Best-effort parse; if it fails, fall back to "now"
            try:
                # Accept either Z-suffixed or offset ISO
                if a.created_at.endswith("Z"):
                    created_dt = datetime.fromisoformat(a.created_at.replace("Z", "+00:00"))
                else:
                    created_dt = datetime.fromisoformat(a.created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                else:
                    created_dt = created_dt.astimezone(timezone.utc)
            except Exception:
                created_dt = datetime.now(timezone.utc)
        else:
            # No timestamp provided → use now
            created_dt = datetime.now(timezone.utc)

        # Persist normalized timestamp back onto the artifact
        a.created_at = created_dt
        # Canonical forms:
        ts_iso = created_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        created_at_ts = created_dt.timestamp()

        # 2) Record in index + occurrence log
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a

        # 2.1) Wire artifact text into ScopedIndices for searchability
        if self.scoped_indices is not None and self.scoped_indices.backend is not None:
            try:
                # Build a simple text blob from labels, metrics, and kind
                # Build text blob
                parts: list[str] = []
                if a.kind:
                    parts.append(str(a.kind))
                if a.tags:
                    parts.extend(a.tags)
                if a.labels:
                    parts.append("; ".join(f"{k}: {v}" for k, v in a.labels.items()))
                if a.metrics:
                    parts.append("; ".join(f"{k}: {v}" for k, v in a.metrics.items()))
                text = " ".join(parts).strip()

                extra_meta: dict[str, Any] = {
                    "run_id": a.run_id,
                    "graph_id": a.graph_id,
                    "node_id": a.node_id,
                    "tool_name": a.tool_name,
                    "tool_version": a.tool_version,
                    "pinned": a.pinned,
                    "tags": a.tags or [],
                }

                # Add artifact kind as explicit metadata field
                if a.kind:
                    extra_meta["artifact_kind"] = a.kind

                # Flatten labels last so they can override if needed
                if a.labels:
                    extra_meta.update(a.labels)

                meta = build_index_meta_from_scope(
                    kind="artifact",
                    source="artifact_index",
                    ts=ts_iso,  # human-readable ISO
                    created_at_ts=created_at_ts,  # numeric timestamp
                    extra=extra_meta,
                )

                await self.scoped_indices.upsert(
                    corpus="artifact",
                    item_id=a.artifact_id,
                    text=text,
                    metadata=meta,
                )
            except Exception:
                import logging

                logging.getLogger("aethergraph.indices").exception(
                    "scoped_indices_artifact_upsert_failed"
                )

        # 3) Metering hook for artifact writes
        try:
            meter = current_metering()

            # Try a few common size fields, fallback to 0
            size = (
                getattr(a, "bytes", None)
                or getattr(a, "size_bytes", None)
                or getattr(a, "size", None)
                or 0
            )

            await meter.record_artifact(
                scope=self.scope,  # Scope carries user/org/run/graph/app/session
                kind=getattr(a, "kind", "unknown"),
                bytes=int(size),
                pinned=bool(getattr(a, "pinned", False)),
            )
        except Exception:
            import logging

            logging.getLogger("aethergraph.metering").exception("record_artifact_failed")

        # 4) Update run/session stores (best-effort; don't break on failure)
        try:
            services = current_services()
        except Exception:
            return  # outside runtime context, nothing to do

        # Update run metadata
        run_store = getattr(services, "run_store", None)
        if run_store is not None and a.run_id:
            record_artifact = getattr(run_store, "record_artifact", None)
            if callable(record_artifact):
                await record_artifact(
                    a.run_id,
                    artifact_id=a.artifact_id,
                    created_at=created_dt,
                )

        # Update session metadata
        session_store = getattr(services, "session_store", None)
        session_id = a.session_id or getattr(self.scope, "session_id", None)
        if session_store is not None and session_id:
            sess_record_artifact = getattr(session_store, "record_artifact", None)
            if callable(sess_record_artifact):
                await sess_record_artifact(
                    session_id,
                    created_at=created_dt,
                )

    # ---------- core staging/ingest ----------
    async def stage_path(self, ext: str = "") -> str:
        """
        Plan a staging file path for artifact creation.

        This method requests a temporary file path from the underlying artifact store,
        suitable for staging a new artifact. The file extension can be specified to
        guide downstream handling (e.g., ".txt", ".json").

        Examples:
            Stage a temporary text file:
            ```python
            staged_path = await context.artifacts().stage_path(".txt")
            ```

            Stage a file with a custom extension:
            ```python
            staged_path = await context.artifacts().stage_path(".log")
            ```

        Args:
            ext: Optional file extension for the staged file (e.g., ".txt", ".json").

        Returns:
            str: The planned staging file path as a string.
        """
        return await self.store.plan_staging_path(planned_ext=ext)

    async def stage_dir(self, suffix: str = "") -> str:
        """
        Plan a staging directory for artifact creation.

        This method requests a temporary directory path from the underlying artifact store,
        suitable for staging a directory artifact. The suffix can be used to distinguish
        different staging contexts.

        Examples:
            Stage a temporary directory:
            ```python
            staged_dir = await context.artifacts().stage_dir()
            ```

            Stage a directory with a custom suffix:
            ```python
            staged_dir = await context.artifacts().stage_dir("_images")
            ```

        Args:
            suffix: Optional string to append to the directory name for uniqueness.

        Returns:
            str: The planned staging directory path as a string.
        """
        return await self.store.plan_staging_dir(suffix=suffix)

    async def ingest_file(
        self,
        staged_path: str,
        *,
        kind: str,
        tags: list[str] | None = None,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        pin: bool = False,
    ) -> Artifact:
        """
        Ingest a staged file as an artifact and record it in the index.

        This method takes a file that has been staged locally, persists it in the
        artifact store, and records its metadata in the artifact index. It supports
        adding labels, metrics, and logical URIs for organization.

        Examples:
            Ingest a staged model file:
            ```python
            artifact = await context.artifacts().ingest_file(
                staged_path="/tmp/model.bin",
                kind="model",
                labels={"domain": "vision"},
                pin=True
            )
            ```

            Ingest with a suggested URI:
            ```python
            artifact = await context.artifacts().ingest_file(
                staged_path="/tmp/data.csv",
                kind="dataset",
                suggested_uri="s3://bucket/data.csv"
            )
            ```

        Args:
            staged_path: The local path to the staged file.
            kind: The artifact type (e.g., "model", "dataset").
            tags: Optional list of tags to associate with the artifact.
            labels: Optional dictionary of metadata labels.
            metrics: Optional dictionary of numeric metrics.
            suggested_uri: Optional logical URI for the artifact.
            pin: If True, pins the artifact for retention.

        Returns:
            Artifact: The fully persisted `Artifact` object with metadata and identifiers.

        Notes:
            The `staged_path` must point to an existing file. The method will handle
            cleanup of the staged file if configured in the underlying store.
            If you already have a file at a specific URI (e.g. "s3://bucket/file" or local file path), consider using `save_file` instead.
        """
        # Start with user labels
        eff_labels: dict[str, Any] = dict(labels or {})

        # Mirror tags into labels for structured search/back-compat
        if tags:
            eff_labels.setdefault("tags", tags)

        # Add scope identity + scope_id
        scoped_labels = self._with_scope_labels(eff_labels)

        a = await self.store.ingest_staged_file(
            staged_path=staged_path,
            kind=kind,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            tags=tags,
            labels=scoped_labels,
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
        Ingest a staged directory as a directory artifact and record it in the index.

        This method takes a directory that has been staged locally, persists its contents
        in the artifact store (optionally creating a manifest or archive), and records
        its metadata in the artifact index. Additional keyword arguments are passed to
        the store's ingest logic.

        Examples:
            Ingest a staged directory with manifest:
            ```python
            artifact = await context.artifacts().ingest_dir(
                staged_dir="/tmp/output_dir",
                kind="directory",
                labels={"type": "images"}
            )
            ```

            Ingest with custom metrics:
            ```python
            artifact = await context.artifacts().ingest_dir(
                staged_dir="/tmp/logs",
                kind="log_dir",
                metrics={"file_count": 12}
            )
            ```

        Args:
            staged_dir: The local path to the staged directory.
            **kwargs: Additional keyword arguments for artifact metadata (e.g., kind, labels, metrics).

        Returns:
            Artifact: The fully persisted `Artifact` object with metadata and identifiers.

        """

        # Extract user labels/tags from kwargs
        raw_labels = kwargs.pop("labels", None)
        tags = kwargs.pop("tags", None)

        eff_labels: dict[str, Any] = dict(raw_labels or {})
        if tags:
            eff_labels.setdefault("tags", tags)

        # Inject scope identity + scope_id
        scoped_labels = self._with_scope_labels(eff_labels)

        # Put back into kwargs for the store
        kwargs["labels"] = scoped_labels
        kwargs["tags"] = tags

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
        tags: list[str] | None = None,
        mime: str | None = None,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        name: str | None = None,
        pin: bool = False,
        cleanup: bool = True,
    ) -> Artifact:
        """
        Save an existing file and index it.

        This method saves a file to the artifact store, associates it with the current
        execution context, and records it in the artifact index. It supports adding
        metadata such as labels, metrics, and a suggested URI for logical organization.

        Examples:
            Basic usage with a file path:
            ```python
            artifact = await context.artifacts().save_file(
                path="/tmp/output.txt",
                kind="text",
                labels={"category": "logs"},
            )
            ```

            Saving a file with a custom name and pinning it:
            ```python
            artifact = await context.artifacts().save_file(
                path="/tmp/data.csv",
                kind="dataset",
                name="data_backup.csv",
                pin=True,
            )
            ```

        Args:
            path: The local file path to save.
            kind: A string representing the artifact type (e.g., "text", "dataset").
            labels: A dictionary of metadata labels to associate with the artifact.
            metrics: A dictionary of numerical metrics to associate with the artifact.
            suggested_uri: A logical URI for the artifact (e.g., "s3://bucket/file").
            name: A custom name for the artifact, used as the `filename` label.
            pin: A boolean indicating whether to pin the artifact.
            cleanup: A boolean indicating whether to delete the local file after saving.

        Returns:
            Artifact: The saved `Artifact` object containing metadata and identifiers.

        Notes:
            The `name` parameter is used to set the `filename` label for the artifact.
            If both `name` and `suggested_uri` are provided, `name` takes precedence for the filename.

        """
        # Start with user labels
        eff_labels: dict[str, Any] = dict(labels or {})

        if tags:
            eff_labels.setdefault("tags", tags)

        # If caller passed an explicit name, prefer that as filename label
        if name:
            eff_labels.setdefault("filename", name)

        # If caller gave a suggested_uri but no explicit name, infer filename from it
        if suggested_uri and "filename" not in eff_labels:
            from pathlib import PurePath

            eff_labels["filename"] = PurePath(suggested_uri).name

        labels = self._with_scope_labels(eff_labels)
        a = await self.store.save_file(
            path=path,
            kind=kind,
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            mime=mime,
            tags=tags,
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
        name: str | None = None,
        kind: str = "text",
        tags: list[str] | None = None,
        labels: dict | None = None,
        metrics: dict | None = None,
        pin: bool = False,
    ) -> Artifact:
        """
        This method stages the text as a temporary `.txt` file, writes the payload,
        and persists it as an artifact with associated metadata. It is accessed via
        `context.artifacts().save_text(...)`.

        Examples:
            Basic usage to save a text artifact:
            ```python
            await context.artifacts().save_text("Hello, world!")
            ```

             Saving with custom metadata and logical filename:
            ```python
            await context.artifacts().save_text(
                "Experiment results",
                name="results.txt",
                labels={"experiment": "A1"},
                metrics={"accuracy": 0.98},
                pin=True
            )
            ```

        Args:
            payload: The text content to be saved as an artifact.
            suggested_uri: Optional logical URI for the artifact. If not provided,
            the `name` will be used if available.
            name: Optional logical filename for the artifact.
            kind: The artifact kind, defaults to `"text"`.
            labels: Optional dictionary of string labels for categorization.
            metrics: Optional dictionary of numeric metrics for tracking.
            pin: If True, pins the artifact for retention.

        Returns:
            Artifact: The fully persisted `Artifact` object containing metadata and storage reference.
        """
        staged = await self.stage_path(".txt")

        def _write() -> str:
            p = Path(staged)
            p.write_text(payload, encoding="utf-8")
            return str(p)

        staged = await asyncio.to_thread(_write)

        # If user gave a logical filename but no suggested_uri, re-use it
        if name and not suggested_uri:
            suggested_uri = name

        return await self.save_file(
            path=staged,
            kind=kind,
            tags=tags,
            labels=labels,
            mime="text/plain",
            metrics=metrics,
            suggested_uri=suggested_uri,
            name=name,
            pin=pin,
        )

    async def save_json(
        self,
        payload: dict,
        *,
        suggested_uri: str | None = None,
        name: str | None = None,
        kind: str = "json",
        tags: list[str] | None = None,
        labels: dict | None = None,
        metrics: dict | None = None,
        pin: bool = False,
    ) -> Artifact:
        """
        Save a JSON payload as an artifact with full context metadata.

        This method stages the JSON data as a temporary `.json` file, writes the payload,
        and persists it as an artifact with associated metadata. It is accessed via
        `context.artifacts().save_json(...)`.

        Examples:
            Basic usage to save a JSON artifact:
            ```python
            await context.artifacts().save_json({"foo": "bar", "count": 42})
            ```

            Saving with custom metadata and logical filename:
            ```python
            await context.artifacts().save_json(
                {"results": [1, 2, 3]},
                name="results.json",
                labels={"experiment": "A1"},
                metrics={"accuracy": 0.98},
                pin=True
            )
            ```

        Args:
            payload: The JSON-serializable dictionary to be saved as an artifact.
            suggested_uri: Optional logical URI for the artifact. If not provided,
                the `name` will be used if available.
            name: Optional logical filename for the artifact.
            kind: The artifact kind, defaults to `"json"`.
            labels: Optional dictionary of string labels for categorization.
            metrics: Optional dictionary of numeric metrics for tracking.
            pin: If True, pins the artifact for retention.

        Returns:
            Artifact: The fully persisted `Artifact` object containing metadata and storage reference.
        """
        staged = await self.stage_path(".json")

        def _write() -> str:
            p = Path(staged)
            import json

            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return str(p)

        staged = await asyncio.to_thread(_write)

        if name and not suggested_uri:
            suggested_uri = name

        return await self.save_file(
            path=staged,
            kind=kind,
            tags=tags,
            mime="application/json",
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            name=name,
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
        Async context manager for streaming artifact writes.

        This method yields a writer object that supports:

        - `writer.write(bytes)` for streaming data
        - `writer.add_labels(...)` to attach metadata
        - `writer.add_metrics(...)` to record metrics

        After the context exits, the writer's artifact is finalized and recorded in the index.
        Accessed via `context.artifacts().writer(...)`.

        Examples:
            Basic usage to stream a file artifact:
            ```python
            async with context.artifacts().writer(kind="binary") as w:
                await w.write(b"some data")
            ```

            Streaming with custom file extension and pinning:
            ```python
            async with context.artifacts().writer(
                kind="log",
                planned_ext=".log",
                pin=True
            ) as w:
                await w.write(b'Log entry 1\\n')
                w.add_labels({"source": 'app'})
                w.add_metrics({"lines": 1})
            ```

        Args:
            kind: The artifact type (e.g., "binary", "log", "text").
            planned_ext: Optional file extension for the staged artifact (e.g., ".txt").
            pin: If True, pins the artifact for retention.

        Returns:
            AsyncIterator[Any]: Yields a writer object for streaming data and metadata.

        Notes:
            - Scope labels are added during `_record` after the context exits, so they are not available during the write phase.
            - If you want tags, call `w.add_labels({"tags": [...]})` inside the context
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

    # ---------- load by artifact ID ----------
    async def get_by_id(self, artifact_id: str) -> Artifact | None:
        """
        Retrieve a single artifact by its unique identifier.

        This asynchronous method queries the configured artifact index for the specified
        `artifact_id`. If the index is not set up, a `RuntimeError` is raised. The method
        is typically accessed via `context.artifacts().get_by_id(...)`.

        Examples:
            Fetching an artifact by ID:
            ```python
            artifact = await context.artifacts().get_by_id("artifact_123")
            if artifact:
                print(artifact.name)
            ```

        Args:
            artifact_id: The unique string identifier of the artifact to retrieve.

        Returns:
            Artifact | None: The matching `Artifact` object if found, otherwise `None`.
        """
        if self.index is None:
            raise RuntimeError("Artifact index is not configured on this facade")
        return await self.index.get(artifact_id)

    async def load_bytes_by_id(self, artifact_id: str) -> bytes:
        """
        Load raw bytes for a file-like artifact by its unique identifier.

        This asynchronous method retrieves the artifact metadata from the index using
        the provided `artifact_id`, then loads the underlying bytes from the artifact store.
        It is accessed via `context.artifacts().load_bytes_by_id(...)`.

        Examples:
            Basic usage to load bytes for an artifact:
            ```python
            data = await context.artifacts().load_bytes_by_id("artifact_123")
            ```

            Handling missing artifacts:
            ```python
            try:
                data = await context.artifacts().load_bytes_by_id("artifact_456")
            except FileNotFoundError:
                print("Artifact not found.")
            ```

        Args:
            artifact_id: The unique string identifier of the artifact to retrieve.

        Returns:
            bytes: The raw byte content of the artifact.

        Raises:
            FileNotFoundError: If the artifact is not found or missing a URI.
        """
        art = await self.get_by_id(artifact_id)
        if art is None or not art.uri:
            raise FileNotFoundError(f"Artifact {artifact_id} not found or missing uri")
        return await self.store.load_artifact_bytes(art.uri)

    async def load_text_by_id(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """
        Load the text content of an artifact by its unique identifier.

        This asynchronous method retrieves the raw bytes for the specified `artifact_id`
        and decodes them into a string using the provided encoding. It is accessed via
        `context.artifacts().load_text_by_id(...)`.

        Examples:
            Basic usage to load text from an artifact:
            ```python
            text = await context.artifacts().load_text_by_id("artifact_123")
            print(text)
            ```

            Loading with custom encoding and error handling:
            ```python
            text = await context.artifacts().load_text_by_id(
                "artifact_456",
                encoding="utf-16",
                errors="ignore"
            )
            ```

        Args:
            artifact_id: The unique string identifier of the artifact to retrieve.
            encoding: The text encoding to use for decoding bytes (default: `"utf-8"`).
            errors: Error handling strategy for decoding (default: `"strict"`).

        Returns:
            str: The decoded text content of the artifact.

        Raises:
            FileNotFoundError: If the artifact is not found or missing a URI.
        """
        data = await self.load_bytes_by_id(artifact_id)
        return data.decode(encoding, errors=errors)

    async def load_json_by_id(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        """
        Load and parse a JSON artifact by its unique identifier.

        This asynchronous method retrieves the raw text content for the specified
        `artifact_id`, decodes it using the provided encoding, and parses it as JSON.
        It is accessed via `context.artifacts().load_json_by_id(...)`.

        Examples:
            Basic usage to load a JSON artifact:
            ```python
            data = await context.artifacts().load_json_by_id("artifact_123")
            print(data["foo"])
            ```

            Loading with custom encoding and error handling:
            ```python
            data = await context.artifacts().load_json_by_id(
                "artifact_456",
                encoding="utf-16",
                errors="ignore"
            )
            ```

        Args:
            artifact_id: The unique string identifier of the artifact to retrieve.
            encoding: The text encoding to use for decoding bytes (default: `"utf-8"`).
            errors: Error handling strategy for decoding (default: `"strict"`).

        Returns:
            Any: The parsed JSON object from the artifact.

        Raises:
            FileNotFoundError: If the artifact is not found or missing a URI.
            json.JSONDecodeError: If the artifact content is not valid JSON.
        """
        text = await self.load_text_by_id(artifact_id, encoding=encoding, errors=errors)
        return json.loads(text)

    async def load_content(
        self,
        artifact_id: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
        max_bytes: int | None = None,
    ) -> ArtifactContent:
        """
        Load artifact content by ID with type-aware decoding.

        Behavior:
        - For JSON artifacts, returns `ArtifactContent(mode="json", json=...)`.
        - For text-like artifacts, returns `ArtifactContent(mode="text", text=...)`.
        - Otherwise returns raw bytes in `ArtifactContent(mode="bytes", data=...)`.

        Examples:
            Load typed content for downstream branching:
            ```python
            content = await context.artifacts().load_content("artifact_123")
            if content.mode == "json":
                print(content.json)
            ```

        Args:
            artifact_id: Artifact identifier from the artifact index.
            encoding: Text decoding encoding used for text/json paths.
            errors: Decode error strategy used for text/json paths.
            max_bytes: Optional byte cap applied only to raw-bytes mode.

        Returns:
            ArtifactContent: Normalized typed content wrapper for the artifact.

        Raises:
            FileNotFoundError: If `artifact_id` is not found in the index.
        """
        art = await self.get_by_id(artifact_id=artifact_id)
        if art is None:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")

        mode = _infer_content_mode(art)

        def maybe_truncate(data: bytes) -> bytes:
            if max_bytes is not None and len(data) > max_bytes:
                return data[:max_bytes]
            return data

        if mode == "json":
            data = await self.load_json_by_id(artifact_id, encoding=encoding, errors=errors)
            return ArtifactContent(artifact=art, mode=mode, json=data)

        if mode == "text":
            text = await self.load_text_by_id(artifact_id, encoding=encoding, errors=errors)
            return ArtifactContent(artifact=art, mode=mode, text=text)

        # raw bytes
        raw = await self.load_bytes_by_id(artifact_id)
        raw = maybe_truncate(raw)
        return ArtifactContent(artifact=art, mode="bytes", data=raw)

    async def as_local_file_by_id(
        self,
        artifact_id: str,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        Resolve an artifact ID to a local file path.

        This resolves the artifact metadata by ID and then delegates to
        `as_local_file(...)` for backend-agnostic local file materialization.

        Examples:
            Resolve by artifact ID:
            ```python
            path = await context.artifacts().as_local_file_by_id("artifact_123")
            ```

        Args:
            artifact_id: Artifact identifier from the artifact index.
            must_exist: If True, validates local path existence/type.

        Returns:
            str: Absolute local file path.
        """
        art = await self.get_by_id(artifact_id)
        if art is None or not art.uri:
            raise FileNotFoundError(f"Artifact {artifact_id} not found or missing uri")
        return await self.as_local_file(art, must_exist=must_exist)

    async def as_local_dir_by_id(
        self,
        artifact_id: str,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        Resolve an artifact ID to a local directory path.

        This resolves the artifact metadata by ID and then delegates to
        `as_local_dir(...)` for backend-agnostic local directory materialization.

        Examples:
            Resolve directory by artifact ID:
            ```python
            path = await context.artifacts().as_local_dir_by_id("artifact_456")
            ```

        Args:
            artifact_id: Artifact identifier from the artifact index.
            must_exist: If True, validates local directory existence.

        Returns:
            str: Absolute local directory path.
        """
        art = await self.get_by_id(artifact_id)
        if art is None or not art.uri:
            raise FileNotFoundError(f"Artifact {artifact_id} not found or missing uri")
        return await self.as_local_dir(art, must_exist=must_exist)

    # ---------- load APIs ----------
    async def load_bytes(self, uri: str) -> bytes:
        """
        Load raw bytes from a file or URI in a backend-agnostic way.

        This method retrieves the byte content from the specified `uri`, supporting both
        local files and remote storage backends. It is accessed via `context.artifacts().load_bytes(...)`.

        Examples:
            Basic usage to load bytes from a local file:
            ```python
            data = await context.artifacts().load_bytes("file:///tmp/model.bin")
            ```

            Loading bytes from an S3 URI:
            ```python
            data = await context.artifacts().load_bytes("s3://bucket/data.bin")
            ```

        Args:
            uri: The URI or path of the file to load. Supports local files and remote storage backends.

        Returns:
            bytes: The raw byte content of the file or artifact.
        """
        return await self.store.load_bytes(uri)

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """
        Load the text content from a file or URI in a backend-agnostic way.

        This method retrieves the raw bytes from the specified `uri`, decodes them into a string
        using the provided encoding, and returns the text. It is accessed via `context.artifacts().load_text(...)`.

        Examples:
            Basic usage to load text from a local file:
            ```python
            text = await context.artifacts().load_text("file:///tmp/output.txt")
            print(text)
            ```

            Loading text from an S3 URI with custom encoding:
            ```python
            text = await context.artifacts().load_text(
                "s3://bucket/data.txt",
                encoding="utf-16"
            )
            ```

        Args:
            uri: The URI or path of the file to load. Supports local files and remote storage backends.
            encoding: The text encoding to use for decoding bytes (default: `"utf-8"`).
            errors: Error handling strategy for decoding (default: `"strict"`).

        Returns:
            str: The decoded text content of the file or artifact.
        """
        return await self.store.load_text(uri, encoding=encoding, errors=errors)

    async def load_json(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        """
        Load and parse a JSON file from the specified URI.

        This asynchronous method retrieves the file contents as text, then parses
        the text into a Python object using the standard `json` library. It is
        typically accessed via `context.artifacts().load_json(...)`.

        Examples:
            Basic usage to load a JSON file:
            ```python
            data = await context.artifacts().load_json("file:///path/to/data.json")
            ```

            Specifying a custom encoding:
            ```python
            data = await context.artifacts().load_json(
                "file:///path/to/data.json",
                encoding="utf-16"
            )
            ```

        Args:
            uri: The URI of the JSON file to load. Supports local and remote paths.
            encoding: The text encoding to use when reading the file (default: "utf-8").
            errors: The error handling scheme for decoding (default: "strict").

        Returns:
            Any: The parsed Python object loaded from the JSON file.
        """
        text = await self.load_text(uri, encoding=encoding, errors=errors)
        return json.loads(text)

    async def load_artifact(self, uri: str) -> Any:
        """
        Compatibility loader for artifact URI.

        This method is retained for backward compatibility and may return bytes
        or a directory path depending on store implementation.

        Examples:
            Load using legacy compatibility API:
            ```python
            value = await context.artifacts().load_artifact(uri)
            ```

        Args:
            uri: Artifact URI.

        Returns:
            Any: Backend-dependent artifact payload.
        """
        warnings.warn(
            "ArtifactFacade.load_artifact() is a compatibility API and will be removed "
            "in a future version. Use explicit load_* methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.store.load_artifact(uri)

    async def load_artifact_bytes(self, uri: str) -> bytes:
        """
        Compatibility byte loader for artifact URI.

        This method is retained for backward compatibility and should be replaced
        with `load_bytes(...)` in new code.

        Examples:
            Load bytes using legacy API:
            ```python
            data = await context.artifacts().load_artifact_bytes(uri)
            ```

        Args:
            uri: Artifact URI.

        Returns:
            bytes: Raw artifact bytes.
        """
        warnings.warn(
            "ArtifactFacade.load_artifact_bytes() is a compatibility API and will be removed "
            "in a future version. Use load_bytes() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.store.load_artifact_bytes(uri)

    async def load_artifact_dir(self, uri: str) -> str:
        """
        Compatibility directory loader for artifact URI.

        This method is retained for backward compatibility and should be replaced
        with `as_local_dir(...)` in new code.

        Examples:
            Resolve directory using legacy API:
            ```python
            path = await context.artifacts().load_artifact_dir(uri)
            ```

        Args:
            uri: Artifact URI.

        Returns:
            str: Local directory path for the artifact.
        """
        warnings.warn(
            "ArtifactFacade.load_artifact_dir() is a compatibility API and will be removed "
            "in a future version. Use as_local_dir() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        This method provides a backend-agnostic way to access directory artifacts as local filesystem paths.
        For local filesystems, it returns the underlying CAS directory. For remote backends (e.g., S3),
        it downloads the directory contents to a staging location and returns the path.

        Examples:
            Basic usage to access a local directory artifact:
            ```python
            local_dir = await context.artifacts().as_local_dir("file:///tmp/output_dir")
            print(local_dir)
            ```

            Handling missing directories:
            ```python
            try:
                local_dir = await context.artifacts().as_local_dir("s3://bucket/data_dir")
            except FileNotFoundError:
                print("Directory not found.")
            ```

        Args:
            artifact_or_uri: The artifact object, URI string, or Path representing the directory.
            must_exist: If True, raises FileNotFoundError if the local path does not exist.

        Returns:
            str: The resolved local filesystem path to the directory artifact.

        Raises:
            FileNotFoundError: If the resolved local directory does not exist and `must_exist` is True.
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
        Resolve an artifact to a local file path.

        This method transparently handles local and remote artifact URIs, downloading
        remote file artifacts to a staging path when needed.

        Examples:
            Using a local file path:
            ```python
            local_path = await context.artifacts().as_local_file("/tmp/data.csv")
            ```

            Using an S3 URI:
            ```python
            local_path = await context.artifacts().as_local_file("s3://bucket/key.csv")
            ```

            Using an Artifact object:
            ```python
            local_path = await context.artifacts().as_local_file(artifact)
            ```

        Args:
            artifact_or_uri: The artifact to resolve, which may be a string URI, Path, or Artifact object.
            must_exist: If True, raises FileNotFoundError if the file does not exist or is not a file.

        Returns:
            str: The absolute path to the local file containing the artifact's data.
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
    async def list(
        self,
        *,
        level: ScopeLevel | None = "run",
        include_node: bool = True,
        tags: list[str] | None = None,
        filters: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        """
        List artifacts using structured index filters.

        Scoping is controlled by `level` (`scope`, `session`, `run`, `user`, `org`)
        and optional node narrowing (`include_node=True` by default).

        Examples:
            List all artifacts for the current run:
            ```python
            artifacts = await context.artifacts().list()
            for a in artifacts:
                print(a.artifact_id, a.kind)
            ```

            List artifacts for the current graph regardless of node:
            ```python
            graph_artifacts = await context.artifacts().list(include_node=False)
            ```

            List artifacts filtered by tags:
            ```python
            tagged = await context.artifacts().list(tags=["report"])
            ```

        Args:
            level: Scope level used to derive tenant/scope/run/session filters.
            include_node: When True, constrain results to current `graph_id` and `node_id`.
            tags: Optional tag filter.
            filters: Extra label filters merged with scope filters.
            limit: Maximum number of rows to return.

        Returns:
            list[Artifact]: Matching artifacts.
        """
        base_labels = self._filters_for_level(level=level)

        # Constrain to this graph/node by default
        if include_node:
            if self.graph_id:
                base_labels.setdefault("graph_id", self.graph_id)
            if self.node_id:
                base_labels.setdefault("node_id", self.node_id)

        eff_labels: dict[str, Any] = dict(filters or {})
        eff_labels.update(base_labels)

        # Tag filter, represented as labels["tags"] for ArtifactIndex
        if tags:
            eff_labels.setdefault("tags", tags)

        return await self.index.search(
            labels=eff_labels or None,
            limit=limit,
        )

    async def search(
        self,
        *,
        query: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        labels: dict[str, str] | None = None,
        metric: str | None = None,
        metric_mode: Literal["max", "min"] | None = None,
        level: ScopeLevel | None = "run",
        extra_scope_labels: dict[str, str] | None = None,
        limit: int | None = None,
        include_graph: bool = False,
        include_node: bool = False,
        time_window: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[Artifact]:
        """
        Search artifacts with structured and optional semantic/lexical retrieval.

        Behavior:
            - If query is None/empty: structured search via ArtifactIndex.
            - If query is non-empty: SearchBackend via ScopedIndices (corpus="artifact"),
              with semantic/lexical/hybrid selected by `mode`.

        Examples:
            Structured search by kind and tags:
            ```python
            rows = await context.artifacts().search(kind="report", tags=["weekly"])
            ```

            Semantic search across artifact index text:
            ```python
            rows = await context.artifacts().search(
                query="model with highest f1",
                mode="semantic",
                limit=10,
            )
            ```

        Args:
            query: Optional free text query for semantic/lexical/hybrid search.
            kind: Optional artifact kind to filter on (structured path only).
            tags: Optional list of tag strings for filtering.
            labels: Extra label filters to apply.
            metric: Optional metric name to optimize (structured path).
            metric_mode: "max" or "min" for structured metric ranking.
            level: Scope level controlling tenant/scope filtering.
            extra_scope_labels: Additional scope labels to merge on top of level filters.
            limit: Maximum number of results (top_k for semantic; limit for structured).
            include_graph: When True, constrain to current graph in addition to `level`.
            include_node: When True, constrain to current node in addition to `level`.
            time_window: Optional time window for created_at_ts filtering (SearchBackend).
            mode: Search mode for the query path ("semantic", "lexical", "hybrid", etc.).

        Returns:
            list[Artifact]: Matching artifacts, preserving search-order for query mode.
        """

        # 1) Build base labels from scope + level (org/user/scope_id/run/session)
        base_labels = self._filters_for_level(level)

        # Optionally constrain by graph/node for structured path
        if include_graph and self.graph_id:
            base_labels.setdefault("graph_id", self.graph_id)
        if include_node and self.node_id:
            base_labels.setdefault("node_id", self.node_id)

        # 2) Merge in user-provided labels and extra scope labels
        eff_labels: dict[str, Any] = dict(labels or {})
        eff_labels.update(base_labels)
        if extra_scope_labels:
            eff_labels.update(extra_scope_labels)

        # 3) Tag filter (stored in labels["tags"] and handled downstream)
        if tags:
            eff_labels.setdefault("tags", tags)

        # --- Structured path: no query or empty query ---------------------
        if not query:
            return await self.index.search(
                kind=kind,
                labels=eff_labels or None,
                metric=metric,
                mode=metric_mode,  # metric ranking mode: "max"/"min"
                limit=limit,
            )

        # --- Semantic / lexical / hybrid path via ScopedIndices ----------
        if self.scoped_indices is None:
            # Fallback: if no SearchBackend is available, degrade to structured search
            return await self.index.search(
                kind=kind,
                labels=eff_labels or None,
                metric=metric,
                mode=metric_mode,
                limit=limit,
            )

        top_k = limit or 20
        eff_mode: SearchMode = mode or "semantic"

        scored = await self.scoped_indices.search(
            corpus="artifact",
            query=query,
            top_k=top_k,
            filters=eff_labels,
            level=None,  # level already applied via eff_labels
            time_window=time_window,
            mode=eff_mode,  # SearchMode, not metric mode
        )

        ids = [s.item_id for s in scored]
        if not ids:
            return []

        arts: list[Artifact] = []
        for art_id in ids:
            art = await self.get_by_id(art_id)
            if art:
                arts.append(art)
        return arts

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        metric_mode: Literal["max", "min"],
        level: ScopeLevel | None = "run",
        tags: list[str] | None = None,
        filters: dict[str, str] | None = None,
    ) -> Artifact | None:
        """
        Return the best artifact for a kind by metric optimization.

        Examples:
            Find the best model by accuracy for the current run:
            ```python
            best_model = await context.artifacts().best(
                kind="model",
                metric="accuracy",
                metric_mode="max"
            )
            ```

            Find the lowest-loss dataset:
            ```python
            best_dataset = await context.artifacts().best(
                kind="dataset",
                metric="loss",
                metric_mode="min",
                level="run"
            )
            ```

            Apply additional label filters:
            ```python
            best_artifact = await context.artifacts().best(
                kind="model",
                metric="f1_score",
                metric_mode="max",
                filters={"domain": "finance"}
            )
            ```

        Args:
            kind: The type of artifact to search for (e.g., "model", "dataset").
            metric: The metric name to optimize (e.g., "accuracy", "loss").
            metric_mode: Optimization mode, either `"max"` or `"min"`.
            level: Scope level controlling tenant/run/session filters.
            tags: Optional tag filter.
            filters: Additional label filters to further restrict the search.

        Returns:
            Artifact | None: The best matching `Artifact` object, or `None` if no match is found.
        """
        eff_filters: dict[str, str] = dict(filters or {})
        eff_filters.update(self._filters_for_level(level))
        if tags:
            eff_filters["tags"] = tags

        return await self.index.best(
            kind=kind,
            metric=metric,
            mode=metric_mode,  # still the metric ranking mode for the index
            filters=eff_filters or None,
        )

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        """
        Mark or unmark an artifact as pinned for retention.

        This asynchronous method updates the `pinned` status of the specified artifact
        in the artifact index. Pinning an artifact ensures it is retained and not subject
        to automatic cleanup. It is accessed via `context.artifacts().pin(...)`.

        Examples:
            Pin an artifact for retention:
            ```python
            await context.artifacts().pin("artifact_123", pinned=True)
            ```

            Unpin an artifact to allow cleanup:
            ```python
            await context.artifacts().pin("artifact_456", pinned=False)
            ```

        Args:
            artifact_id: The unique string identifier of the artifact to update.
            pinned: Boolean indicating whether to pin (`True`) or unpin (`False`) the artifact.

        Returns:
            None
        """
        await self.index.pin(artifact_id, pinned=pinned)

    # ---------- search result hydration ----------
    async def fetch_artifacts_for_search_results(
        self,
        scored_items: list[ScoredItem],
        corpus: str = "artifact",
    ) -> list[ArtifactSearchResult]:
        """
        Hydrate scored search hits into typed artifact search results.

        This converts generic `ScoredItem` entries into `ArtifactSearchResult`
        objects by loading artifact metadata for matching corpus items.

        Examples:
            Hydrate scoped-indices search output:
            ```python
            hydrated = await context.artifacts().fetch_artifacts_for_search_results(scored)
            ```

        Args:
            scored_items: Raw scored search rows from search backend.
            corpus: Corpus name to include (default: `"artifact"`).

        Returns:
            list[ArtifactSearchResult]: Hydrated artifact search result objects.
        """
        artifact_items = [it for it in scored_items if it.corpus == corpus]
        results: list[ArtifactSearchResult] = []
        for it in artifact_items:
            art = await self.get_by_id(it.item_id)
            results.append(ArtifactSearchResult(item=it, artifact=art))
        return results

    # ---------- internal helpers ----------
    async def _record_simple(self, a: Artifact) -> None:
        """
        Record artifact metadata in index without extra side effects.

        This is an internal lightweight variant of `_record(...)` that only
        updates index and occurrence log.

        Examples:
            Record minimal index state:
            ```python
            await facade._record_simple(artifact)
            ```

        Args:
            a: Artifact to upsert.

        Returns:
            None: Updates index/occurrence state.
        """
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a

    # ---------- deprecated / compatibility ----------
    async def stage(self, ext: str = "") -> str:
        """
        Deprecated compatibility alias for `stage_path()`.

        This method will be removed in a future version.

        Examples:
            Legacy usage:
            ```python
            staged = await context.artifacts().stage(".txt")
            ```

        Args:
            ext: Optional extension passed to `stage_path(...)`.

        Returns:
            str: Planned staging file path.
        """
        warnings.warn(
            "ArtifactFacade.stage() is deprecated and will be removed in a future version. "
            "Use stage_path().",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """
        Deprecated compatibility alias for `ingest_file()`.

        This method will be removed in a future version.

        Examples:
            Legacy usage:
            ```python
            a = await context.artifacts().ingest("/tmp/file", kind="text")
            ```

        Args:
            staged_path: Staged local file path.
            kind: Artifact kind.
            labels: Optional labels dictionary.
            metrics: Optional metrics dictionary.
            suggested_uri: Optional suggested URI/name.
            pin: Whether to pin artifact.

        Returns:
            Artifact: Persisted artifact record.
        """
        warnings.warn(
            "ArtifactFacade.ingest() is deprecated and will be removed in a future version. "
            "Use ingest_file().",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """
        Deprecated compatibility alias for `save_file()`.

        This method will be removed in a future version.

        Examples:
            Legacy usage:
            ```python
            a = await context.artifacts().save("/tmp/file", kind="text")
            ```

        Args:
            path: Source file path.
            kind: Artifact kind.
            labels: Optional labels dictionary.
            metrics: Optional metrics dictionary.
            suggested_uri: Optional suggested URI/name.
            pin: Whether to pin artifact.

        Returns:
            Artifact: Persisted artifact record.
        """
        warnings.warn(
            "ArtifactFacade.save() is deprecated and will be removed in a future version. "
            "Use save_file().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.save_file(
            path,
            kind=kind,
            labels=labels,
            metrics=metrics,
            suggested_uri=suggested_uri,
            pin=pin,
        )

    async def tmp_path(self, suffix: str = "") -> str:  # DEPRECATED: use stage_path()
        """
        Deprecated compatibility alias for `stage_path()`.

        This method will be removed in a future version.

        Examples:
            Legacy usage:
            ```python
            staged = await context.artifacts().tmp_path(".txt")
            ```

        Args:
            suffix: Suffix/extension forwarded to `stage_path(...)`.

        Returns:
            str: Planned staging file path.
        """
        warnings.warn(
            "ArtifactFacade.tmp_path() is deprecated and will be removed in a future version. "
            "Use stage_path().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.stage_path(suffix)

    # FS-only, legacy helpers — prefer as_local_dir/as_local_file for new code
    def to_local_path(
        self,
        uri_or_path: str | Path | Artifact,
        *,
        must_exist: bool = True,
    ) -> str:
        """
        Deprecated FS-only path resolver.

        This assumes local filesystem paths and does not correctly materialize
        remote backends. Use `as_local_file(...)` / `as_local_dir(...)` instead.

        Examples:
            Legacy local path resolution:
            ```python
            p = context.artifacts().to_local_path("file:///tmp/a.txt")
            ```

        Args:
            uri_or_path: Local path/URI or Artifact.
            must_exist: Validate local existence when True.

        Returns:
            str: Local path string (or raw URI for non-file schemes).
        """
        warnings.warn(
            "ArtifactFacade.to_local_path() is deprecated and will be removed in a future version. "
            "Use as_local_file()/as_local_dir().",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """
        Deprecated FS-only file resolver.

        Use `await as_local_file(...)` for backend-agnostic behavior.

        Examples:
            Legacy local file resolution:
            ```python
            p = context.artifacts().to_local_file("/tmp/a.txt")
            ```

        Args:
            uri_or_path: Local path/URI or Artifact.
            must_exist: Validate local existence/type when True.

        Returns:
            str: Local file path.
        """
        warnings.warn(
            "ArtifactFacade.to_local_file() is deprecated and will be removed in a future version. "
            "Use as_local_file().",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """
        Deprecated FS-only directory resolver.

        Use `await as_local_dir(...)` for backend-agnostic behavior.

        Examples:
            Legacy local directory resolution:
            ```python
            p = context.artifacts().to_local_dir("/tmp/mydir")
            ```

        Args:
            uri_or_path: Local path/URI or Artifact.
            must_exist: Validate local existence/type when True.

        Returns:
            str: Local directory path.
        """
        warnings.warn(
            "ArtifactFacade.to_local_dir() is deprecated and will be removed in a future version. "
            "Use as_local_dir().",
            DeprecationWarning,
            stacklevel=2,
        )
        p = Path(self.to_local_path(uri_or_path, must_exist=must_exist))
        if must_exist and not p.is_dir():
            raise NotADirectoryError(f"Expected directory, got file: {p}")
        return str(p)
