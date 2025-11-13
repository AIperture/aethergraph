from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union
from contextlib import asynccontextmanager
from aethergraph.contracts.services.artifacts import AsyncArtifactStore, AsyncArtifactIndex, Artifact
from .paths import _from_uri_or_path
from urllib.parse import urlparse
from pathlib import Path

Scope = Literal["node", "run", "graph", "all"]

class ArtifactFacade:
    """ Facade for artifact storage and indexing operations within a specific context.
        Provides async methods to stage, ingest, save, and write artifacts with automatic indexing.
    """
    def __init__(self, *, run_id: str, graph_id: str, node_id: str,
                 tool_name: str, tool_version: str,
                 store: AsyncArtifactStore, index: AsyncArtifactIndex):
        self.run_id, self.graph_id, self.node_id = run_id, graph_id, node_id
        self.tool_name, self.tool_version = tool_name, tool_version
        self.store, self.index = store, index
        self.last_artifact: Optional[Artifact] = None

    async def stage(self, ext: str = "") -> str:
        return await self.store.plan_staging_path(ext)

    async def ingest(self, staged_path: str, *, kind: str, labels=None, metrics=None,
                     suggested_uri: Optional[str]=None, pin: bool=False):
        a = await self.store.ingest_staged_file(
            staged_path=staged_path, kind=kind,
            run_id=self.run_id, graph_id=self.graph_id, node_id=self.node_id,
            tool_name=self.tool_name, tool_version=self.tool_version,
            labels=labels, metrics=metrics, suggested_uri=suggested_uri, pin=pin
        )
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        return a

    async def save(self, path: str, *, kind: str, labels=None, metrics=None,
                   suggested_uri: Optional[str]=None, pin: bool=False):
        a = await self.store.save_file(
            path=path, kind=kind,
            run_id=self.run_id, graph_id=self.graph_id, node_id=self.node_id,
            tool_name=self.tool_name, tool_version=self.tool_version,
            labels=labels, metrics=metrics, suggested_uri=suggested_uri, pin=pin
        )
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a
        return a

    async def save_text(self, payload: str, *, suggested_uri: Optional[str] = None):
        a = await self.store.save_text(
            payload=payload, suggested_uri=suggested_uri
        )
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a
        return a
    
    async def save_json(self, payload: dict, *, suggested_uri: Optional[str] = None):
        a = await self.store.save_json(
            payload=payload, suggested_uri=suggested_uri
        )
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a
        return a
    
    @asynccontextmanager
    async def writer(self, *, kind: str, planned_ext: str | None = None, pin: bool = False):
        # Use the store's (sync) contextmanager via async wrapper; user writes bytes
        cm = await self.store.open_writer(
            kind=kind, run_id=self.run_id, graph_id=self.graph_id, node_id=self.node_id,
            tool_name=self.tool_name, tool_version=self.tool_version,
            planned_ext=planned_ext, pin=pin
        )
        with cm as w:
            yield w
            a = getattr(w, "_artifact", None)
        if a:
            await self.index.upsert(a)
            await self.index.record_occurrence(a)
            self.last_artifact = a
        else:
            self.last_artifact = None

    async def stage_dir(self, suffix: str = "") -> str:
        return await self.store.plan_staging_dir(suffix)

    async def ingest_dir(self, staged_dir: str, **kw):
        a = await self.store.ingest_directory(
            staged_dir=staged_dir, run_id=self.run_id, graph_id=self.graph_id,
            node_id=self.node_id, tool_name=self.tool_name, tool_version=self.tool_version,
            **kw
        )
        await self.index.upsert(a)
        await self.index.record_occurrence(a)
        self.last_artifact = a
        return a

    async def tmp_path(self, suffix: str = "") -> str:
        return await self.store.plan_staging_path(suffix)

    async def load_bytes(self, uri: str) -> bytes:
        return await self.store.load_bytes(uri)
    
    async def load_text(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        data = await self.store.load_text(uri)
        return data
    
    async def load_json(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> Any:
        data = await self.store.load_json(uri, encoding=encoding, errors=errors)
        return data

    async def load_artifact(self, uri: str) -> Any:
        return await self.store.load_artifact(uri)

    async def load_artifact_bytes(self, uri: str) -> bytes:
        return await self.store.load_artifact_bytes(uri)

    # ------- indexing pass-throughs with scoping -------
    async def list(self, *, scope: Scope="run") -> List[Artifact]:
        """
        Quick listing scoped to current run/graph/node by default.
        scope:
          - "node": filter by (run_id, graph_id, node_id)
          - "graph": filter by (run_id, graph_id)
          - "run": filter by (run_id)   [default]
          - "all": no implicit filters (dangerous; use sparingly)
        """
        if scope == "node":
            arts = await self.index.search(labels={"graph_id": self.graph_id, "node_id": self.node_id})
            return [a for a in arts if a.run_id == self.run_id]
        if scope == "graph":
            arts = await self.index.search(labels={"graph_id": self.graph_id})
            return [a for a in arts if a.run_id == self.run_id]
        if scope == "run":
            return await self.index.list_for_run(self.run_id)
        if scope == "all":
            return await self.index.search()
        return await self.index.search(labels=self._scope_labels(scope))

    async def search(
        self,
        *,
        kind: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
        metric: Optional[str] = None,
        mode: Optional[Literal["max", "min"]] = None,
        scope: Scope = "run",
        extra_scope_labels: Optional[Dict[str, Any]] = None,
    ) -> List[Artifact]:
        """Pass-through search with automatic scoping."""
        eff_labels = dict(labels or {})
        if scope in ("node", "graph", "project"):
            eff_labels.update(self._scope_labels(scope))
        if extra_scope_labels:
            eff_labels.update(extra_scope_labels)
        # Delegate heavy lifting to the index
        return await self.index.search(kind=kind, labels=eff_labels, metric=metric, mode=mode)

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        scope: Scope = "run",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Artifact]:
        eff_filters = dict(filters or {})
        if scope in ("node", "graph", "project"):
            eff_filters.update(self._scope_labels(scope))
        return await self.index.best(kind=kind, metric=metric, mode=mode, filters=eff_filters or None)

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        await self.index.pin(artifact_id, pinned)

    # -------- internal helpers --------
    def _scope_labels(self, scope: Scope) -> Dict[str, Any]:
        if scope == "node":
            return {"run_id": self.run_id, "graph_id": self.graph_id, "node_id": self.node_id}
        if scope == "graph":
            return {"run_id": self.run_id, "graph_id": self.graph_id}
        if scope == "run":
            return {"run_id": self.run_id}
        return {}  # "all"

    def _project_id(self) -> Optional[str]:
        # This function is no longer used, but kept for possible future use.
        return getattr(self, "project_id", None)
    

    # ---------- convenience: URI -> local path (FS only) ----------
    def to_local_path(self, uri_or_path: Union[str, Path, Artifact], *, must_exist: bool = True) -> str:
        """
        Return an absolute native path string if input is a file:// URI or local path.
        If given an Artifact, uses artifact.uri.
        If the scheme is not file://, returns the string form unchanged (or raise in strict mode).
        """
        if isinstance(uri_or_path, Artifact):
            s = uri_or_path.uri or ""
        else:
            s = str(uri_or_path)

        p = _from_uri_or_path(s).resolve()

        # If not a file:// (e.g., s3://, http://), _from_uri_or_path returns Path(s);
        # detect that and either pass through or raise for clarity.
        u = urlparse(s)
        if "://" in s and (u.scheme or "").lower() != "file":
            # Not a filesystem artifact; caller likely needs a downloader
            return s  # or: raise ValueError("Not a local filesystem URI")

        if must_exist and not p.exists():
            raise FileNotFoundError(f"Local path not found: {p}")
        return str(p)

    def to_local_file(self, uri_or_path: Union[str, Path, Artifact], *, must_exist: bool = True) -> str:
        """Same as to_local_path but asserts it's a file (not a dir)."""
        p = Path(self.to_local_path(uri_or_path, must_exist=must_exist))
        if must_exist and not p.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {p}")
        return str(p)

    def to_local_dir(self, uri_or_path: Union[str, Path, Artifact], *, must_exist: bool = True) -> str:
        """Same as to_local_path but asserts it's a directory."""
        p = Path(self.to_local_path(uri_or_path, must_exist=must_exist))
        if must_exist and not p.is_dir():
            raise NotADirectoryError(f"Expected directory, got file: {p}")
        return str(p)