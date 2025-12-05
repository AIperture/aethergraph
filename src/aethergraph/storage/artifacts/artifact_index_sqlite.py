from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from typing import Any, Literal

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex


class SqliteArtifactIndexSync:
    """
    SQLite-based artifact index.

    - Good for tens/hundreds of thousands of artifacts.
    - Stores labels/metrics as JSON.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT,
                graph_id TEXT,
                node_id TEXT,
                tool_name TEXT,
                tool_version TEXT,
                kind TEXT,
                sha256 TEXT,
                bytes INTEGER,
                mime TEXT,
                created_at TEXT,
                labels_json TEXT,
                metrics_json TEXT,
                pinned INTEGER DEFAULT 0
                -- NOTE: older DBs created without uri/preview_uri columns
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifact_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT,
                run_id TEXT,
                graph_id TEXT,
                node_id TEXT,
                tool_name TEXT,
                tool_version TEXT,
                created_at TEXT,
                labels_json TEXT
            )
            """
        )

        # Migration: add uri / preview_uri columns if missing
        cur.execute("PRAGMA table_info(artifacts)")
        cols = {row["name"] for row in cur.fetchall()}

        if "uri" not in cols:
            cur.execute("ALTER TABLE artifacts ADD COLUMN uri TEXT")
        if "preview_uri" not in cols:
            cur.execute("ALTER TABLE artifacts ADD COLUMN preview_uri TEXT")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(sha256)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_occ_artifact ON artifact_occurrences(artifact_id)"
        )
        self._conn.commit()

    def upsert(self, a: Artifact) -> None:
        rec = a.to_dict()
        labels_json = json.dumps(rec.get("labels") or {}, ensure_ascii=False)
        metrics_json = json.dumps(rec.get("metrics") or {}, ensure_ascii=False)

        self._conn.execute(
            """
            INSERT INTO artifacts (
                artifact_id, run_id, graph_id, node_id,
                tool_name, tool_version, kind, sha256,
                bytes, mime, created_at, labels_json, metrics_json,
                pinned, uri, preview_uri
            ) VALUES (
                :artifact_id, :run_id, :graph_id, :node_id,
                :tool_name, :tool_version, :kind, :sha256,
                :bytes, :mime, :created_at, :labels_json, :metrics_json,
                :pinned, :uri, :preview_uri
            )
            ON CONFLICT(artifact_id) DO UPDATE SET
                run_id        = excluded.run_id,
                graph_id      = excluded.graph_id,
                node_id       = excluded.node_id,
                tool_name     = excluded.tool_name,
                tool_version  = excluded.tool_version,
                kind          = excluded.kind,
                sha256        = excluded.sha256,
                bytes         = excluded.bytes,
                mime          = excluded.mime,
                created_at    = excluded.created_at,
                labels_json   = excluded.labels_json,
                metrics_json  = excluded.metrics_json,
                pinned        = excluded.pinned,
                uri           = excluded.uri,
                preview_uri   = excluded.preview_uri
            """,
            {
                "artifact_id": rec["artifact_id"],
                "run_id": rec["run_id"],
                "graph_id": rec["graph_id"],
                "node_id": rec["node_id"],
                "tool_name": rec["tool_name"],
                "tool_version": rec["tool_version"],
                "kind": rec["kind"],
                "sha256": rec["sha256"],
                "bytes": rec["bytes"],
                "mime": rec["mime"],
                "created_at": rec["created_at"],
                "labels_json": labels_json,
                "metrics_json": metrics_json,
                "pinned": int(rec.get("pinned") or 0),
                "uri": rec.get("uri"),
                "preview_uri": rec.get("preview_uri"),
            },
        )
        self._conn.commit()

    def list_for_run(self, run_id: str) -> list[Artifact]:
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        )
        rows = cur.fetchall()
        return [self._row_to_artifact(r) for r in rows]

    def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, Any] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        where = []
        params: list[Any] = []

        if kind:
            where.append("kind = ?")
            params.append(kind)

        # label filters: naive JSON LIKE for now (can be improved later)
        if labels:
            for k, v in labels.items():
                where.append("labels_json LIKE ?")
                # crude but works: `"k": "v"` substring
                params.append(f'%"{k}": "{v}"%')

        order_by = "created_at ASC"
        if metric and mode:
            # We'll compute metrics ordering in Python after fetch for simplicity.
            order_by = "created_at ASC"

        sql = "SELECT * FROM artifacts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_by}"

        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        cur = self._conn.execute(sql, params)
        rows = [self._row_to_artifact(r) for r in cur.fetchall()]

        if metric and mode:
            rows = [a for a in rows if metric in (a.metrics or {})]
            rows.sort(
                key=lambda a: a.metrics[metric],
                reverse=(mode == "max"),
            )
            if limit is not None:
                rows = rows[:limit]

        return rows

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
        self._conn.execute(
            "UPDATE artifacts SET pinned = ? WHERE artifact_id = ?",
            (int(bool(pinned)), artifact_id),
        )
        self._conn.commit()

    def record_occurrence(self, a: Artifact, extra_labels: dict | None = None) -> None:
        labels = {**(a.labels or {}), **(extra_labels or {})}
        labels_json = json.dumps(labels, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO artifact_occurrences (
                artifact_id, run_id, graph_id, node_id,
                tool_name, tool_version, created_at, labels_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                a.artifact_id,
                a.run_id,
                a.graph_id,
                a.node_id,
                a.tool_name,
                a.tool_version,
                a.created_at,
                labels_json,
            ),
        )
        self._conn.commit()

    # -------- helpers --------

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        labels = json.loads(row["labels_json"] or "{}")
        metrics = json.loads(row["metrics_json"] or "{}")
        return Artifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            graph_id=row["graph_id"],
            node_id=row["node_id"],
            tool_name=row["tool_name"],
            tool_version=row["tool_version"],
            kind=row["kind"],
            sha256=row["sha256"],
            bytes=row["bytes"],
            mime=row["mime"],
            created_at=row["created_at"],
            labels=labels,
            metrics=metrics,
            pinned=bool(row["pinned"]),
            uri=row["uri"],  #  real URI
            preview_uri=row["preview_uri"],  # real preview URI (may be None)
        )

    def get(self, artifact_id: str) -> Artifact | None:
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        row = cur.fetchone()
        if row:
            return self._row_to_artifact(row)
        return None


class SqliteArtifactIndex(AsyncArtifactIndex):
    def __init__(self, path: str):
        self._sync = SqliteArtifactIndexSync(path)

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
