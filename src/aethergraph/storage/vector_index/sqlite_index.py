from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pickle
import sqlite3
import threading
from typing import Any

import numpy as np

from aethergraph.contracts.storage.vector_index import VectorIndex

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


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


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (n, d)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _l2_normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


class SQLiteVectorIndex(VectorIndex):
    """
    Simple SQLite-backed vector index.

    Baseline path uses brute-force cosine similarity over SQL-limited candidates. :contentReference[oaicite:1]{index=1}

    Optional FAISS acceleration:
      - If faiss is installed and enabled, maintains a per-corpus FAISS HNSW index on disk.
      - Index is marked dirty on add/delete and rebuilt lazily on next search. NOTE: this can be slow for large corpora.
      - This is a local index for small to medium workloads; for distributed or large-scale use cases, consider other backends.

    Promoted fields you *may* pass in meta: :contentReference[oaicite:2]{index=2}
      - scope_id, user_id, org_id, client_id, session_id
      - run_id, graph_id, node_id
      - kind, source
      - created_at_ts (float UNIX timestamp)
    """

    def __init__(
        self,
        root: str,
        *,
        use_faiss_if_available: bool = True,
        faiss_m: int = 32,  # HNSW M
        faiss_ef_search: int = 64,  # query-time accuracy/speed knob
        faiss_ef_construction: int = 200,  # build-time accuracy/speed knob
        faiss_probe_factor: int = 20,  # fetch k * factor candidates then post-filter
        faiss_probe_min: int = 200,
        faiss_probe_max: int = 5000,
        brute_force_candidate_limit: int = 5000,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.db_path = str(self.root / "index.sqlite")
        _ensure_db(self.db_path)

        # --- FAISS config ---
        self._faiss_enabled = bool(use_faiss_if_available and faiss is not None)
        self._faiss_dir = self.root / "faiss"
        self._faiss_dir.mkdir(parents=True, exist_ok=True)

        self._faiss_m = int(faiss_m)
        self._faiss_ef_search = int(faiss_ef_search)
        self._faiss_ef_construction = int(faiss_ef_construction)
        self._faiss_probe_factor = int(faiss_probe_factor)
        self._faiss_probe_min = int(faiss_probe_min)
        self._faiss_probe_max = int(faiss_probe_max)

        self._brute_force_candidate_limit = int(brute_force_candidate_limit)

        self._faiss_lock = threading.RLock()
        self._faiss_cache: dict[str, tuple[Any, list[str], int]] = {}
        # cache: corpus_id -> (faiss_index, id_to_chunk_id, dim)
        self._faiss_dirty: set[str] = set()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _faiss_paths(self, corpus_id: str) -> tuple[Path, Path]:
        safe = corpus_id.replace("/", "_")
        return (self._faiss_dir / f"{safe}.index", self._faiss_dir / f"{safe}.meta.pkl")

    def _mark_dirty(self, corpus_id: str) -> None:
        if not self._faiss_enabled:
            return
        with self._faiss_lock:
            self._faiss_dirty.add(corpus_id)
            self._faiss_cache.pop(corpus_id, None)

    def _build_faiss_index_from_db(self, corpus_id: str) -> tuple[Any, list[str], int]:
        """
        Build an HNSW cosine index for all vectors in a corpus.

        We normalize vectors and use inner product (IP) => cosine similarity.
        """
        if not self._faiss_enabled:
            raise RuntimeError("FAISS is not enabled/available.")

        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT chunk_id, vec FROM embeddings WHERE corpus_id=?", (corpus_id,))
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            # empty corpus index
            dim = 0
            index = None
            return index, [], dim

        # Infer dim from first vector
        first_vec = np.frombuffer(rows[0][1], dtype=np.float32)
        dim = int(first_vec.shape[0])

        # Base HNSW (IP metric), wrapped with ID map so ids are stable ints
        base = faiss.IndexHNSWFlat(dim, self._faiss_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = self._faiss_ef_construction
        base.hnsw.efSearch = self._faiss_ef_search
        index = faiss.IndexIDMap2(base)

        id_to_chunk: list[str] = []
        next_id = 0

        # Add in batches to keep memory reasonable
        batch_size = 2048
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            chunk_ids = [r[0] for r in batch]
            mats = [np.frombuffer(r[1], dtype=np.float32) for r in batch]
            x = np.stack(mats, axis=0).astype(np.float32, copy=False)
            x = _l2_normalize_rows(x)

            ids = np.arange(next_id, next_id + len(chunk_ids), dtype=np.int64)
            index.add_with_ids(x, ids)

            id_to_chunk.extend(chunk_ids)
            next_id += len(chunk_ids)

        return index, id_to_chunk, dim

    def _ensure_faiss_ready(self, corpus_id: str) -> tuple[Any, list[str], int]:
        if not self._faiss_enabled:
            raise RuntimeError("FAISS is not enabled/available.")

        with self._faiss_lock:
            cached = self._faiss_cache.get(corpus_id)
            if cached is not None and corpus_id not in self._faiss_dirty:
                return cached

            index_path, meta_path = self._faiss_paths(corpus_id)

            # If not dirty and files exist, load from disk
            if corpus_id not in self._faiss_dirty and index_path.exists() and meta_path.exists():
                index = faiss.read_index(str(index_path))
                with meta_path.open("rb") as f:
                    meta = pickle.load(f)
                id_to_chunk = meta["id_to_chunk"]
                dim = int(meta["dim"])
                # Ensure query-time params applied
                try:
                    index.index.hnsw.efSearch = self._faiss_ef_search  # type: ignore[attr-defined]
                except Exception:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning("Failed to set faiss efSearch parameter.")

                self._faiss_cache[corpus_id] = (index, id_to_chunk, dim)
                return index, id_to_chunk, dim

            # Otherwise rebuild from DB
            index, id_to_chunk, dim = self._build_faiss_index_from_db(corpus_id)

            # Persist (if non-empty)
            if index is not None:
                faiss.write_index(index, str(index_path))
                with meta_path.open("wb") as f:
                    pickle.dump({"id_to_chunk": id_to_chunk, "dim": dim}, f)

            self._faiss_dirty.discard(corpus_id)
            self._faiss_cache[corpus_id] = (index, id_to_chunk, dim)
            return index, id_to_chunk, dim

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

            # Mark FAISS corpus index dirty (lazy rebuild on next search)
            self._mark_dirty(corpus_id)

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

            # Mark FAISS corpus index dirty (lazy rebuild on next search)
            self._mark_dirty(corpus_id)

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

    def _passes_where(self, where: dict[str, Any], row: dict[str, Any]) -> bool:
        # where only applies to promoted fields in your SQL path :contentReference[oaicite:3]{index=3}
        # Here, we enforce the same semantics post-hoc.
        for k, v in where.items():
            if v is None:
                continue
            if row.get(k) != v:
                return False
        return True

    def _fetch_rows_for_chunk_ids(
        self,
        conn: sqlite3.Connection,
        corpus_id: str,
        chunk_ids: list[str],
    ) -> dict[str, tuple[str, float | None]]:
        """
        Returns {chunk_id: (meta_json, created_at_ts)} for given chunk_ids.
        Batched to avoid SQLite parameter limits.
        """
        out: dict[str, tuple[str, float | None]] = {}
        cur = conn.cursor()

        # SQLite default param limit is often 999, keep headroom
        batch_size = 900
        for i in range(0, len(chunk_ids), batch_size):
            b = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in b)
            sql = f"""
                SELECT e.chunk_id, c.meta_json, e.created_at_ts
                FROM embeddings e
                JOIN chunks c
                ON e.corpus_id = c.corpus_id AND e.chunk_id = c.chunk_id
                WHERE e.corpus_id = ?
                AND e.chunk_id IN ({placeholders})
            """
            cur.execute(sql, [corpus_id, *b])
            for cid, meta_json, created_at_ts in cur.fetchall():
                out[str(cid)] = (str(meta_json), created_at_ts)
        return out

    def _search_bruteforce_sync(
        self,
        corpus_id: str,
        q: np.ndarray,
        k: int,
        where: dict[str, Any],
        max_candidates: int | None,
        created_at_min: float | None,
        created_at_max: float | None,
    ) -> list[dict[str, Any]]:
        # This is your original SQL-candidate path (kept as a fallback). :contentReference[oaicite:4]{index=4}
        qn = float(np.linalg.norm(q) + 1e-9)

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

            for key, val in where.items():
                if val is None:
                    continue
                if key in promoted_cols:
                    sql += f" AND e.{key} = ?"
                    params.append(val)

            if created_at_min is not None:
                sql += " AND e.created_at_ts >= ?"
                params.append(created_at_min)
            if created_at_max is not None:
                sql += " AND e.created_at_ts <= ?"
                params.append(created_at_max)

            candidate_limit = max_candidates or self._brute_force_candidate_limit
            sql += " ORDER BY e.created_at_ts DESC"
            sql += " LIMIT ?"
            params.append(candidate_limit)

            cur.execute(sql, params)
            rows = cur.fetchall()
        finally:
            conn.close()

        # Minor speedup: avoid json.loads until after top-k
        scored: list[tuple[float, str, str]] = []
        for chunk_id, vec_bytes, norm, meta_json in rows:
            v = np.frombuffer(vec_bytes, dtype=np.float32)
            score = float(np.dot(q, v) / (qn * (norm or 1e-9)))
            scored.append((score, str(chunk_id), str(meta_json)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        out: list[dict[str, Any]] = []
        for score, chunk_id, meta_json in top:
            out.append({"chunk_id": chunk_id, "score": score, "meta": json.loads(meta_json)})
        return out

    def _search_faiss_sync(
        self,
        corpus_id: str,
        q: np.ndarray,
        k: int,
        where: dict[str, Any],
        created_at_min: float | None,
        created_at_max: float | None,
        max_candidates: int | None,
    ) -> list[dict[str, Any]]:
        # If max_candidates is set very small, caller probably expects strict recency-bounded behavior.
        # In that case, keep the old semantics.
        if max_candidates is not None and max_candidates <= self._brute_force_candidate_limit:
            return self._search_bruteforce_sync(
                corpus_id=corpus_id,
                q=q,
                k=k,
                where=where,
                max_candidates=max_candidates,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )

        index, id_to_chunk, dim = self._ensure_faiss_ready(corpus_id)
        if index is None or dim <= 0 or not id_to_chunk:
            return []

        if q.shape[0] != dim:
            # Dim mismatch: fall back to brute-force rather than throwing.
            return self._search_bruteforce_sync(
                corpus_id=corpus_id,
                q=q,
                k=k,
                where=where,
                max_candidates=max_candidates,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )

        qn = _l2_normalize_vec(q.astype(np.float32, copy=False))
        qn = qn.reshape(1, -1)

        # Probe progressively deeper until we have k results that pass filters
        probe = max(self._faiss_probe_min, k * self._faiss_probe_factor)
        probe = min(probe, self._faiss_probe_max)

        conn = self._connect()
        try:
            while True:
                scores, ids = index.search(qn, probe)
                ids0 = ids[0]
                scores0 = scores[0]

                # Map to chunk_ids in rank order
                ranked_chunk_ids: list[str] = []
                ranked_scores: list[float] = []
                for fid, sc in zip(ids0, scores0, strict=False):
                    if fid < 0:
                        continue
                    if fid >= len(id_to_chunk):
                        continue
                    ranked_chunk_ids.append(id_to_chunk[int(fid)])
                    ranked_scores.append(float(sc))

                if not ranked_chunk_ids:
                    return []

                # Fetch metas/timestamps for these candidates in batch
                row_map = self._fetch_rows_for_chunk_ids(conn, corpus_id, ranked_chunk_ids)

                out: list[dict[str, Any]] = []
                for cid, sc in zip(ranked_chunk_ids, ranked_scores, strict=True):
                    tup = row_map.get(cid)
                    if tup is None:
                        continue
                    meta_json, created_at_ts = tup
                    meta = json.loads(meta_json)

                    # Apply where/time filters post-hoc
                    if where and not self._passes_where(where, meta):
                        continue
                    if created_at_min is not None:  # noqa: SIM102
                        if created_at_ts is None or float(created_at_ts) < float(created_at_min):
                            continue
                    if created_at_max is not None:  # noqa: SIM102
                        if created_at_ts is None or float(created_at_ts) > float(created_at_max):
                            continue

                    out.append({"chunk_id": cid, "score": sc, "meta": meta})
                    if len(out) >= k:
                        return out

                # Not enough after filtering -> probe deeper or fall back
                if probe >= self._faiss_probe_max:
                    # If filters are too tight, FAISS post-filtering may not find enough.
                    # Fall back to SQL-candidate brute-force which is exact under filters.
                    return self._search_bruteforce_sync(
                        corpus_id=corpus_id,
                        q=q,
                        k=k,
                        where=where,
                        max_candidates=max_candidates,
                        created_at_min=created_at_min,
                        created_at_max=created_at_max,
                    )

                probe = min(self._faiss_probe_max, probe * 2)
        finally:
            conn.close()

    async def search(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
        where: dict[str, Any] | None = None,
        max_candidates: int | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[dict[str, Any]]:
        q = np.asarray(query_vec, dtype=np.float32)
        where = where or {}

        def _search_sync() -> list[dict[str, Any]]:
            if self._faiss_enabled:
                return self._search_faiss_sync(
                    corpus_id=corpus_id,
                    q=q,
                    k=k,
                    where=where,
                    created_at_min=created_at_min,
                    created_at_max=created_at_max,
                    max_candidates=max_candidates,
                )
            return self._search_bruteforce_sync(
                corpus_id=corpus_id,
                q=q,
                k=k,
                where=where,
                max_candidates=max_candidates,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )

        return await asyncio.to_thread(_search_sync)
