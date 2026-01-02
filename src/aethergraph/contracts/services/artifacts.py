from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Artifact:
    artifact_id: str
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    kind: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    mime: str | None = None
    created_at: str | None = None
    labels: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    pinned: bool = False
    uri: str | None = None
    preview_uri: str | None = None
    # tenant fields
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "kind": self.kind,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "mime": self.mime,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "created_at": self.created_at,
            "labels": self.labels,
            "metrics": self.metrics,
            "preview_uri": self.preview_uri,
            "pinned": self.pinned,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "client_id": self.client_id,
            "app_id": self.app_id,
            "session_id": self.session_id,
        }


class AsyncArtifactStore(Protocol):
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
        cleanup: bool = True,
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
    async def load_artifact(self, uri: str) -> Any: ...
    async def load_artifact_bytes(self, uri: str) -> bytes: ...
    async def load_artifact_dir(self, uri: str) -> str: ...
    async def cleanup_tmp(self, max_age_hours: int = 24) -> None: ...
    async def save_text(self, payload: str, suggested_uri: str | None = None) -> Artifact: ...
    async def save_json(self, payload: dict, suggested_uri: str | None = None) -> Artifact: ...
    @property
    def base_uri(self) -> str: ...


class AsyncArtifactIndex(Protocol):
    async def upsert(self, a: Artifact) -> None: ...
    async def list_for_run(self, run_id: str) -> list[Artifact]: ...
    async def search(
        self,
        *,
        kind: str | None = None,
        labels: dict | None = None,
        metric: str | None = None,
        mode: str | None = None,
        limit: int | None = None,
    ) -> list[Artifact]: ...
    async def best(
        self, *, kind: str, metric: str, mode: str, filters: dict | None = None
    ) -> Artifact | None: ...
    async def pin(self, artifact_id: str, pinned: bool = True) -> None: ...
    async def record_occurrence(self, a: Artifact, extra_labels: dict | None = None) -> None: ...
