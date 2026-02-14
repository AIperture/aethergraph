from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from aethergraph.contracts.storage.vector_index import PROMOTED_FIELDS

LEXICAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS fts_docs (
    corpus_id     TEXT,
    chunk_id      TEXT,
    meta_json     TEXT,
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

CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs_idx
USING fts5(
    corpus_id,
    chunk_id,
    text,
    tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_fts_docs_corpus_scope_time
    ON fts_docs(corpus_id, scope_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_fts_docs_corpus_user_time
    ON fts_docs(corpus_id, user_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_fts_docs_corpus_org_time
    ON fts_docs(corpus_id, org_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_fts_docs_corpus_kind_time
    ON fts_docs(corpus_id, kind, created_at_ts DESC);
"""


def _normalize_fts_query(text: str) -> str:
    """
    Normalize a raw user query into a simple FTS5-safe expression.

    Strategy:
      - lowercase
      - extract word-ish tokens (letters/digits/underscore)
      - join with spaces (implicit AND in FTS5)

    This deliberately *removes* punctuation like "?", ":", etc.,
    which often cause FTS syntax errors if left as-is.
    """
    if not text:
        return ""

    # Lowercase, then keep only "word" tokens
    lowered = text.lower()
    tokens = re.findall(r"[0-9A-Za-z_]+", lowered)

    if not tokens:
        return ""

    return " ".join(tokens)


def _ensure_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        cur = conn.cursor()
        for stmt in LEXICAL_SCHEMA.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                cur.execute(s)
        conn.commit()
    finally:
        conn.close()


class SQLiteLexicalIndex:
    """
    Simple SQLite FTS5-based lexical index.

    - Stores text + metadata in `fts_docs`.
    - Uses `fts_docs_idx` (fts5) for MATCH queries.
    - Supports the same promoted-field filters and time bounds as VectorIndex.

    This is intentionally minimal and mirrors the search contract of
    GenericSearchBackend's LexicalIndex protocol.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.db_path = str(self.root / "lexical.sqlite")
        _ensure_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]],
    ) -> None:
        if not chunk_ids:
            return

        def _add_sync():
            conn = self._connect()
            try:
                cur = conn.cursor()
                for cid, text, meta in zip(chunk_ids, texts, metas, strict=True):
                    text = text or ""
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

                    # docs table
                    cur.execute(
                        """
                        REPLACE INTO fts_docs(
                            corpus_id,
                            chunk_id,
                            meta_json,
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
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            corpus_id,
                            cid,
                            meta_json,
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

                    # FTS table
                    cur.execute(
                        """
                        REPLACE INTO fts_docs_idx(corpus_id, chunk_id, text)
                        VALUES (?, ?, ?)
                        """,
                        (corpus_id, cid, text),
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
                        f"DELETE FROM fts_docs WHERE corpus_id=? AND chunk_id IN ({placeholders})",
                        [corpus_id, *chunk_ids],
                    )
                    cur.execute(
                        f"DELETE FROM fts_docs_idx WHERE corpus_id=? AND chunk_id IN ({placeholders})",
                        [corpus_id, *chunk_ids],
                    )
                else:
                    cur.execute("DELETE FROM fts_docs WHERE corpus_id=?", (corpus_id,))
                    cur.execute("DELETE FROM fts_docs_idx WHERE corpus_id=?", (corpus_id,))
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_delete_sync)

    async def search(
        self,
        corpus_id: str,
        query: str,
        k: int,
        index_filters: dict[str, Any] | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lexical search using FTS5.

        - query: free text; we normalize to a safe FTS expression.
        - index_filters: promoted fields (scope_id, user_id, etc.) as scalars.
        - created_at_*: recency bounds.
        """
        index_filters = index_filters or {}

        # --- Normalize user query into a safe FTS expression ----------------
        fts_expr = _normalize_fts_query(query)
        if not fts_expr:
            # Nothing usable → no lexical matches
            if self.debug:
                print(
                    "SQLiteLexicalIndex.search: empty FTS expression after normalization; returning no results."
                )
            return []

        # Escape single quotes so the literal stays valid SQL
        fts_expr_sql = fts_expr.replace("'", "''")

        def _search_sync() -> list[dict[str, Any]]:
            conn = self._connect()
            try:
                cur = conn.cursor()

                # NOTE: fts_expr_sql is inlined as a literal – no MATCH ? placeholder.
                sql = f"""
                    SELECT d.chunk_id,
                        d.meta_json,
                        d.created_at_ts,
                        bm25(fts_docs_idx) AS score
                    FROM fts_docs_idx
                    JOIN fts_docs AS d
                    ON d.corpus_id = fts_docs_idx.corpus_id
                    AND d.chunk_id  = fts_docs_idx.chunk_id
                    WHERE fts_docs_idx MATCH '{fts_expr_sql}'
                    AND d.corpus_id = ?
                """
                params: list[Any] = [corpus_id]

                promoted_cols = set(PROMOTED_FIELDS)

                for key, val in index_filters.items():
                    if val is None:
                        continue
                    if key in promoted_cols:
                        sql += f" AND d.{key} = ?"
                        params.append(val)

                if created_at_min is not None:
                    sql += " AND d.created_at_ts >= ?"
                    params.append(created_at_min)
                if created_at_max is not None:
                    sql += " AND d.created_at_ts <= ?"
                    params.append(created_at_max)

                sql += " ORDER BY score LIMIT ?"
                params.append(k)

                cur.execute(sql, params)
                rows = cur.fetchall()
            finally:
                conn.close()

            out: list[dict[str, Any]] = []
            for chunk_id, meta_json, _created_at_ts, score in rows:
                meta = json.loads(meta_json or "{}")
                out.append(
                    {
                        "chunk_id": str(chunk_id),
                        "score": float(score),
                        "meta": meta,
                    }
                )
            return out

        return await asyncio.to_thread(_search_sync)
