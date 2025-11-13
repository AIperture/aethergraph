from __future__ import annotations
import os, json, sqlite3, math
from typing import List, Dict, Any
import numpy as np
from .base import VectorIndex

"""A simple SQLite-based vector index per corpus (brute-force cosine similarity).
Stores vectors as BLOBs along with metadata in a simple schema.
"""


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

def _ensure_db(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    def __init__(self, index_path: str):
        super().__init__(index_path)
        self.db_path = os.path.join(index_path, "index.sqlite")
        _ensure_db(self.db_path)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    async def add(self, corpus_id: str, chunk_ids: List[str], vectors: List[list[float]], metas: List[Dict[str,Any]]):

        conn = self._connect()
        try:
            cur = conn.cursor()
            for cid, vec, meta in zip(chunk_ids, vectors, metas):
                v = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(v) + 1e-9)
                cur.execute("REPLACE INTO chunks(corpus_id,chunk_id,meta_json) VALUES(?,?,?)",
                            (corpus_id, cid, json.dumps(meta, ensure_ascii=False)))
                cur.execute("REPLACE INTO embeddings(corpus_id,chunk_id,vec,norm) VALUES(?,?,?,?)",
                            (corpus_id, cid, v.tobytes(), norm))
            conn.commit()
        finally:
            conn.close()

    async def delete(self, corpus_id: str, chunk_ids: List[str] | None = None):
        conn = self._connect()
        try:
            cur = conn.cursor()
            if chunk_ids:
                q = f"DELETE FROM chunks WHERE corpus_id=? AND chunk_id IN ({','.join(['?']*len(chunk_ids))})"
                cur.execute(q, [corpus_id, *chunk_ids])
                q2 = f"DELETE FROM embeddings WHERE corpus_id=? AND chunk_id IN ({','.join(['?']*len(chunk_ids))})"
                cur.execute(q2, [corpus_id, *chunk_ids])
            else:
                cur.execute("DELETE FROM chunks WHERE corpus_id=?", (corpus_id,))
                cur.execute("DELETE FROM embeddings WHERE corpus_id=?", (corpus_id,))
            conn.commit()
        finally:
            conn.close()

    async def list_chunks(self, corpus_id: str) -> List[str]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT chunk_id FROM chunks WHERE corpus_id=?", (corpus_id,))
            return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    async def search(self, corpus_id: str, query_vec: list[float], k: int) -> List[Dict[str,Any]]:        # Brute-force cosine similarity. Loads vectors for that corpus.
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q) + 1e-9)

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT e.chunk_id, e.vec, e.norm, c.meta_json FROM embeddings e JOIN chunks c USING(corpus_id,chunk_id) WHERE e.corpus_id=?", (corpus_id,))
            rows = cur.fetchall()
        finally:
            conn.close()

        scores = []
        for chunk_id, vec_bytes, norm, meta_json in rows:
            v = np.frombuffer(vec_bytes, dtype=np.float32)
            score = float(np.dot(q, v) / (qn * norm))
            scores.append((score, chunk_id, meta_json))

        scores.sort(reverse=True, key=lambda x: x[0])
        top = scores[:k]
        out = []
        for score, chunk_id, meta_json in top:
            out.append({"chunk_id": chunk_id, "score": score, "meta": json.loads(meta_json)})
        return out