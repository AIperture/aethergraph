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
    corpus_id TEXT,
    chunk_id  TEXT,
    vec       BLOB,    -- np.float32 array bytes
    norm      REAL,
    PRIMARY KEY (corpus_id, chunk_id)
);
"""


def _ensure_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        for stmt in SCHEMA.strip().split(";\n"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
    finally:
        conn.close()


class SQLiteVectorIndex(VectorIndex):
    """
    Simple SQLite-backed vector index.
    Uses brute-force cosine similarity per corpus.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.root / "index.sqlite")
        _ensure_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        # Each call gets its own connection: thread-safe with to_thread.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return conn

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
                    cur.execute(
                        "REPLACE INTO chunks(corpus_id,chunk_id,meta_json) VALUES(?,?,?)",
                        (corpus_id, cid, json.dumps(meta, ensure_ascii=False)),
                    )
                    cur.execute(
                        "REPLACE INTO embeddings(corpus_id,chunk_id,vec,norm) VALUES(?,?,?,?)",
                        (corpus_id, cid, v.tobytes(), norm),
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
                score = float(np.dot(q, v) / (qn * norm))
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
