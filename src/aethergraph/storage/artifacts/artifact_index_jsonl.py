from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Literal

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex


class JsonlArtifactIndexSync:
    """
    Simple JSONL-based artifact index for small/medium scale.
    Not suitable for millions of artifacts due to linear scans.
    """

    def __init__(self, path: str, occurrences_path: str | None = None):
        self.path = path
        self.occ_path = occurrences_path or (os.path.splitext(path)[0] + "_occurrences.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._by_id: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    self._by_id[rec["artifact_id"]] = rec

    # -------- core operations --------

    def upsert(self, a: Artifact) -> None:
        with self._lock:
            rec = a.to_dict()
            self._by_id[a.artifact_id] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def list_for_run(self, run_id: str) -> list[Artifact]:
        return [Artifact(**r) for r in self._by_id.values() if r.get("run_id") == run_id]

    def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, Any] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Artifact]:
        # NOTE: JSONL index keeps all artifacts in memory (_by_id.values()) and
        # performs filtering / sorting in Python, then applies offset + limit.
        # This is intended for small/medium local installs and tests only.
        rows = list(self._by_id.values())

        if kind:
            rows = [r for r in rows if r.get("kind") == kind]

        # Treat tenant keys as top-level fields, not labels
        TENANT_KEYS = {
            "org_id",
            "user_id",
            "client_id",
            "app_id",
            "session_id",
            "run_id",
        }

        if labels:
            for k, v in labels.items():
                if k in TENANT_KEYS:
                    # Match against top-level JSON fields
                    rows = [r for r in rows if r.get(k) == v]
                    continue

                # Normal label filters
                if isinstance(v, list):
                    rows = [
                        r
                        for r in rows
                        if isinstance(r.get("labels", {}).get(k), list)
                        and set(v).issubset(set(r["labels"][k]))
                    ]
                else:
                    rows = [r for r in rows if r.get("labels", {}).get(k) == v]

        if metric and mode:
            rows = [r for r in rows if metric in r.get("metrics", {})]
            rows.sort(
                key=lambda r: r["metrics"][metric],
                reverse=(mode == "max"),
            )

        if offset > 0:
            rows = rows[offset:]

        if limit is not None:
            rows = rows[:limit]

        return [Artifact(**r) for r in rows]

    def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        filters: dict[str, Any] | None = None,
    ) -> Artifact | None:
        rows = self.search(
            kind=kind,
            labels=filters,
            metric=metric,
            mode=mode,
            limit=1,
        )
        return rows[0] if rows else None

    def pin(self, artifact_id: str, pinned: bool = True) -> None:
        with self._lock:
            if artifact_id not in self._by_id:
                return
            rec = self._by_id[artifact_id]
            rec["pinned"] = bool(pinned)
            self._by_id[artifact_id] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def record_occurrence(self, a: Artifact, extra_labels: dict | None = None) -> None:
        row = {
            "artifact_id": a.artifact_id,
            "run_id": a.run_id,
            "graph_id": a.graph_id,
            "node_id": a.node_id,
            "tool_name": a.tool_name,
            "tool_version": a.tool_version,
            "created_at": a.created_at,
            "labels": {**(a.labels or {}), **(extra_labels or {})},
        }
        with open(self.occ_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def get(self, artifact_id: str) -> Artifact | None:
        if artifact_id in self._by_id:
            return Artifact(**self._by_id[artifact_id])
        return None


class JsonlArtifactIndex(AsyncArtifactIndex):
    """Async wrapper for JsonlArtifactIndexSync using asyncio.to_thread."""

    def __init__(self, path: str, occurrences_path: str | None = None):
        self._sync = JsonlArtifactIndexSync(path, occurrences_path)

    async def upsert(self, a: Artifact) -> None:
        await asyncio.to_thread(self._sync.upsert, a)

    async def list_for_run(self, run_id: str) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.list_for_run, run_id)

    async def search(self, **kwargs) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.search, **kwargs)

    async def best(self, **kwargs) -> Artifact | None:
        return await asyncio.to_thread(self._sync.best, **kwargs)

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        await asyncio.to_thread(self._sync.pin, artifact_id, pinned)

    async def record_occurrence(self, a: Artifact, extra_labels: dict | None = None) -> None:
        await asyncio.to_thread(self._sync.record_occurrence, a, extra_labels)

    async def get(self, artifact_id: str) -> Artifact | None:
        return await asyncio.to_thread(self._sync.get, artifact_id)
