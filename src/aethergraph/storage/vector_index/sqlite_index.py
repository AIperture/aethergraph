from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from aethergraph.contracts.storage.vector_index import VectorIndex

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    corpus_id TEXT,
    chunk_id  TEXT,
    meta_json TEXT,
    PRIMARY KEY (corpus_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    corpus_id     TEXT,
    chunk_id      TEXT,
    vec           BLOB,    -- np.float32 array bytes
    norm          REAL,
    -- promoted / hot fields
    scope_id      TEXT,
    user_id       TEXT,
    org_id        TEXT,
    client_id     TEXT,
    session_id    TEXT,
    run_id        TEXT,
    graph_id      TEXT,
    node_id       TEXT,
    kind          TEXT,
    source        TEXT,
    created_at_ts REAL,
    PRIMARY KEY (corpus_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_emb_corpus_scope_time
    ON embeddings(corpus_id, scope_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_emb_corpus_user_time
    ON embeddings(corpus_id, user_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_emb_corpus_org_time
    ON embeddings(corpus_id, org_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_emb_corpus_kind_time
    ON embeddings(corpus_id, kind, created_at_ts DESC);
"""


def _ensure_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        cur = conn.cursor()
        for stmt in SCHEMA.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                cur.execute(s)
        conn.commit()
    finally:
        conn.close()


class SQLiteVectorIndex(VectorIndex):
    """
    Simple SQLite-backed vector index.

    Uses brute-force cosine similarity per corpus.

    Promoted fields you *may* pass in meta:
      - scope_id, user_id, org_id, client_id, session_id
      - run_id, graph_id, node_id
      - kind, source
      - created_at_ts (float UNIX timestamp)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.root / "index.sqlite")
        _ensure_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metas: list[dict[str, Any]],
    ) -> None:
        if not chunk_ids:
            return

        def _add_sync():
            conn = self._connect()
            try:
                cur = conn.cursor()
                for cid, vec, meta in zip(chunk_ids, vectors, metas, strict=True):
                    v = np.asarray(vec, dtype=np.float32)
                    norm = float(np.linalg.norm(v) + 1e-9)

                    meta_json = json.dumps(meta, ensure_ascii=False)

                    # promoted, optional
                    scope_id = meta.get("scope_id")
                    user_id = meta.get("user_id")
                    org_id = meta.get("org_id")
                    client_id = meta.get("client_id")
                    session_id = meta.get("session_id")
                    run_id = meta.get("run_id")
                    graph_id = meta.get("graph_id")
                    node_id = meta.get("node_id")
                    kind = meta.get("kind")
                    source = meta.get("source")
                    created_at_ts = meta.get("created_at_ts")

                    cur.execute(
                        "REPLACE INTO chunks(corpus_id,chunk_id,meta_json) VALUES(?,?,?)",
                        (corpus_id, cid, meta_json),
                    )
                    cur.execute(
                        """
                        REPLACE INTO embeddings(
                            corpus_id,
                            chunk_id,
                            vec,
                            norm,
                            scope_id,
                            user_id,
                            org_id,
                            client_id,
                            session_id,
                            run_id,
                            graph_id,
                            node_id,
                            kind,
                            source,
                            created_at_ts
                        )
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            corpus_id,
                            cid,
                            v.tobytes(),
                            norm,
                            scope_id,
                            user_id,
                            org_id,
                            client_id,
                            session_id,
                            run_id,
                            graph_id,
                            node_id,
                            kind,
                            source,
                            created_at_ts,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_add_sync)

    async def delete(self, corpus_id: str, chunk_ids: list[str] | None = None) -> None:
        def _delete_sync():
            conn = self._connect()
            try:
                cur = conn.cursor()
                if chunk_ids:
                    placeholders = ",".join("?" for _ in chunk_ids)
                    cur.execute(
                        f"DELETE FROM chunks WHERE corpus_id=? AND chunk_id IN ({placeholders})",
                        [corpus_id, *chunk_ids],
                    )
                    cur.execute(
                        f"DELETE FROM embeddings WHERE corpus_id=? AND chunk_id IN ({placeholders})",
                        [corpus_id, *chunk_ids],
                    )
                else:
                    cur.execute("DELETE FROM chunks WHERE corpus_id=?", (corpus_id,))
                    cur.execute("DELETE FROM embeddings WHERE corpus_id=?", (corpus_id,))
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_delete_sync)

    async def list_chunks(self, corpus_id: str) -> list[str]:
        def _list_sync() -> list[str]:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT chunk_id FROM chunks WHERE corpus_id=?", (corpus_id,))
                return [r[0] for r in cur.fetchall()]
            finally:
                conn.close()

        return await asyncio.to_thread(_list_sync)

    async def list_corpora(self) -> list[str]:
        def _list_sync() -> list[str]:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT corpus_id FROM chunks")
                return [r[0] for r in cur.fetchall()]
            finally:
                conn.close()

        return await asyncio.to_thread(_list_sync)

    async def search(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
        where: dict[str, Any] | None = None,
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q) + 1e-9)

        where = where or {}

        def _search_sync() -> list[dict[str, Any]]:
            conn = self._connect()
            try:
                cur = conn.cursor()

                sql = """
                    SELECT e.chunk_id, e.vec, e.norm, c.meta_json
                    FROM embeddings e
                    JOIN chunks c
                      ON e.corpus_id = c.corpus_id AND e.chunk_id = c.chunk_id
                    WHERE e.corpus_id=?
                """
                params: list[Any] = [corpus_id]

                # Promoted columns we can filter on
                promoted_cols = {
                    "scope_id",
                    "user_id",
                    "org_id",
                    "client_id",
                    "session_id",
                    "run_id",
                    "graph_id",
                    "node_id",
                    "kind",
                    "source",
                }

                # Add equality predicates for known promoted columns
                for key, val in where.items():
                    if val is None:
                        continue
                    if key in promoted_cols:
                        sql += f" AND e.{key} = ?"
                        params.append(val)

                # Bound candidate count and use time ordering to hit recent stuff first.
                # This plays nicely with your idx_emb_corpus_*_time indexes.
                # Fallback: if no max_candidates, use some sane upper bound.
                candidate_limit = max_candidates or 5000
                sql += " ORDER BY e.created_at_ts DESC"
                sql += " LIMIT ?"
                params.append(candidate_limit)

                cur.execute(sql, params)
                rows = cur.fetchall()
            finally:
                conn.close()

            scored: list[tuple[float, str, dict[str, Any]]] = []
            for chunk_id, vec_bytes, norm, meta_json in rows:
                v = np.frombuffer(vec_bytes, dtype=np.float32)
                score = float(np.dot(q, v) / (qn * (norm or 1e-9)))
                meta = json.loads(meta_json)
                scored.append((score, chunk_id, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:k]

            out: list[dict[str, Any]] = []
            for score, chunk_id, meta in top:
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "score": score,
                        "meta": meta,
                    }
                )
            return out

        return await asyncio.to_thread(_search_sync)

    async def search_old(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q) + 1e-9)

        def _search_sync() -> list[dict[str, Any]]:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT e.chunk_id, e.vec, e.norm, c.meta_json
                    FROM embeddings e
                    JOIN chunks c
                      ON e.corpus_id = c.corpus_id AND e.chunk_id = c.chunk_id
                    WHERE e.corpus_id=?
                    """,
                    (corpus_id,),
                )
                rows = cur.fetchall()
            finally:
                conn.close()

            scored: list[tuple[float, str, dict[str, Any]]] = []
            for chunk_id, vec_bytes, norm, meta_json in rows:
                v = np.frombuffer(vec_bytes, dtype=np.float32)
                score = float(np.dot(q, v) / (qn * (norm or 1e-9)))
                meta = json.loads(meta_json)
                scored.append((score, chunk_id, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:k]

            out: list[dict[str, Any]] = []
            for score, chunk_id, meta in top:
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "score": score,
                        "meta": meta,
                    }
                )
            return out

        return await asyncio.to_thread(_search_sync)
