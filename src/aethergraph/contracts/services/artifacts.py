from __future__ import annotations
from typing import Protocol, Optional, Dict, Any, ContextManager
from dataclasses import dataclass

@dataclass
class Artifact:
    artifact_id: str
    uri: str
    kind: str
    bytes: int
    sha256: str
    mime: Optional[str]
    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str
    created_at: str
    labels: Dict[str, Any]
    metrics: Dict[str, Any]
    preview_uri: Optional[str] = None # for rendering previews in UI, not tied to storage
    pinned: bool = False

    def to_dict(self) -> Dict[str, Any]:
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
        }


class AsyncArtifactStore(Protocol):
    async def save_file(self, *, path: str, kind: str, run_id: str, graph_id: str, node_id: str,
                        tool_name: str, tool_version: str, suggested_uri: Optional[str]=None,
                        pin: bool=False, labels: Optional[dict]=None, metrics: Optional[dict]=None,
                        preview_uri: Optional[str]=None) -> Artifact: ...
    async def open_writer(self, *, kind: str, run_id: str, graph_id: str, node_id: str,
                          tool_name: str, tool_version: str, planned_ext: Optional[str]=None,
                          pin: bool=False) -> ContextManager[Any]: ...
    async def plan_staging_path(self, planned_ext: str = "") -> str: ...
    async def ingest_staged_file(self, *, staged_path: str, kind: str, run_id: str, graph_id: str,
                                 node_id: str, tool_name: str, tool_version: str, pin: bool=False,
                                 labels: Optional[dict]=None, metrics: Optional[dict]=None,
                                 preview_uri: Optional[str]=None, suggested_uri: Optional[str]=None) -> Artifact: ...
    async def plan_staging_dir(self, suffix: str = "") -> str: ...
    async def ingest_directory(self, *, staged_dir: str, kind: str, run_id: str, graph_id: str,
                               node_id: str, tool_name: str, tool_version: str,
                               include: Optional[list[str]]=None, exclude: Optional[list[str]]=None,
                               index_children: bool=False, pin: bool=False, labels: Optional[dict]=None,
                               metrics: Optional[dict]=None, suggested_uri: Optional[str]=None,
                               archive: bool=False, archive_name: str="bundle.tar.gz",
                               cleanup: bool=True, store: Optional[str]=None) -> Artifact: ...
    async def load_artifact(self, uri: str) -> Any: ...
    async def load_artifact_bytes(self, uri: str) -> bytes: ...
    async def load_artifact_dir(self, uri: str) -> str: ...
    async def cleanup_tmp(self, max_age_hours: int = 24) -> None: ...
    async def save_text(self, payload: str, suggested_uri: Optional[str]=None) -> Artifact: ...
    async def save_json(self, payload: dict, suggested_uri: Optional[str]=None) -> Artifact: ...
    @property
    def base_uri(self) -> str: ...

class AsyncArtifactIndex(Protocol):
    async def upsert(self, a: Artifact) -> None: ...
    async def list_for_run(self, run_id: str) -> list[Artifact]: ...
    async def search(self, *, kind: Optional[str]=None, labels: Optional[dict]=None,
                     metric: Optional[str]=None, mode: Optional[str]=None) -> list[Artifact]: ...
    async def best(self, *, kind: str, metric: str, mode: str, filters: Optional[dict]=None) -> Optional[Artifact]: ...
    async def pin(self, artifact_id: str, pinned: bool=True) -> None: ...
    async def record_occurrence(self, a: Artifact, extra_labels: dict|None=None) -> None: ...
