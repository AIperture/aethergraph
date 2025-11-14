# aethergraph/artifacts/index_jsonl.py
from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Literal

from aethergraph.contracts.services.artifacts import Artifact


class JsonlArtifactIndexSync:
    """Simple JSONL-based artifact index for small to medium scale use cases.
    Not suitable for very large scale (millions of artifacts) due to linear scans.
    """

    def __init__(self, path: str, occurrences_path: str | None = None):
        self.path = path
        self.occ_path = occurrences_path or (os.path.splitext(path)[0] + "_occurrences.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # small in-memory map for quick lookup / dedup of last write
        self._by_id = {}
        self._lock = threading.Lock()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    self._by_id[rec["artifact_id"]] = rec

    def upsert(self, a: Artifact) -> None:
        """Upsert an artifact record."""
        with self._lock:
            rec = a.to_dict()
            self._by_id[a.artifact_id] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def list_for_run(self, run_id: str) -> list[Artifact]:
        """List all artifacts for a given run_id."""
        return [Artifact(**r) for r in self._by_id.values() if r.get("run_id") == run_id]

    def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, str] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
    ) -> list[Artifact]:
        """Search artifacts by kind, labels (exact match), and metric (min/max)."""
        rows = list(self._by_id.values())
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        if labels:
            for k, v in labels.items():
                rows = [r for r in rows if r.get("labels", {}).get(k) == v]
        if metric and mode:
            rows = [r for r in rows if metric in r.get("metrics", {})]
            rows.sort(key=lambda r: r["metrics"][metric], reverse=(mode == "max"))
        return [Artifact(**r) for r in rows]

    def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        filters: dict[str, str] | None = None,
    ) -> Artifact | None:
        """Get the best artifact by metric with optional filters."""
        rows = self.search(kind=kind, labels=filters, metric=metric, mode=mode)
        return rows[0] if rows else None

    def pin(self, artifact_id: str, pinned: bool = True) -> None:
        """Pin or unpin an artifact by artifact_id."""
        if artifact_id in self._by_id:
            self._by_id[artifact_id]["pinned"] = bool(pinned)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._by_id[artifact_id]) + "\n")

    def record_occurrence(self, a: Artifact, extra_labels: dict | None = None):
        """
        Append-only log that this artifact appeared in this run/node at this time.
        Keeps lineage even if bytes are identical across runs.
        """
        row = {
            "artifact_id": a.artifact_id,
            "run_id": a.run_id,
            "graph_id": a.graph_id,
            "node_id": a.node_id,
            "tool_name": a.tool_name,
            "tool_version": a.tool_version,
            "created_at": a.created_at,
            "labels": a.labels | (extra_labels or {}),
        }
        with open(self.occ_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


class JsonlArtifactIndex:  # implements AsyncArtifactIndex
    def __init__(self, path: str, occurrences_path: str | None = None):
        self._sync = JsonlArtifactIndexSync(path, occurrences_path)

    async def upsert(self, a: Artifact) -> None:
        await asyncio.to_thread(self._sync.upsert, a)

    async def list_for_run(self, run_id: str) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.list_for_run, run_id)

    async def search(self, **kw) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.search, **kw)

    async def best(self, **kw) -> Artifact | None:
        return await asyncio.to_thread(self._sync.best, **kw)

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        await asyncio.to_thread(self._sync.pin, artifact_id, pinned)

    async def record_occurrence(self, a: Artifact, extra_labels: dict | None = None):
        await asyncio.to_thread(self._sync.record_occurrence, a, extra_labels)
