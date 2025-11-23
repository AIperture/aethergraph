from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from aethergraph.contracts.services.artifacts import Artifact


class AsyncArtifactStore(Protocol):
    @property
    def base_uri(self) -> str: ...

    # ---------- save / ingest ----------
    async def save_file(
        self,
        *,
        path: str,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        suggested_uri: str | None = None,
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        preview_uri: str | None = None,
    ) -> Artifact: ...

    async def open_writer(
        self,
        *,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        planned_ext: str | None = None,
        pin: bool = False,
    ) -> AbstractAsyncContextManager[Any]: ...

    async def plan_staging_path(self, planned_ext: str = "") -> str: ...
    async def ingest_staged_file(
        self,
        *,
        staged_path: str,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        preview_uri: str | None = None,
        suggested_uri: str | None = None,
    ) -> Artifact: ...

    async def plan_staging_dir(self, suffix: str = "") -> str: ...
    async def ingest_directory(
        self,
        *,
        staged_dir: str,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        index_children: bool = False,
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        archive: bool = False,
        archive_name: str = "bundle.tar.gz",
        cleanup: bool = True,
        store: str | None = None,
    ) -> Artifact: ...

    # ---------- load ----------
    async def load_bytes(self, uri: str) -> bytes: ...
    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str: ...

    async def load_artifact(self, uri: str) -> Any: ...
    async def load_artifact_bytes(self, uri: str) -> bytes: ...
    async def load_artifact_dir(self, uri: str) -> str: ...

    # ---------- housekeeping ----------
    async def cleanup_tmp(self, max_age_hours: int = 24) -> None: ...
