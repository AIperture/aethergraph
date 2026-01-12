from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend

LEXICAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    corpus_id     TEXT,
    item_id       TEXT,
    text          TEXT,
    meta_json     TEXT,
    created_at_ts REAL,
    org_id        TEXT,
    user_id       TEXT,
    scope_id      TEXT,
    client_id     TEXT,
    app_id        TEXT,
    session_id    TEXT,
    run_id        TEXT,
    graph_id      TEXT,
    node_id       TEXT,
    kind          TEXT,
    source        TEXT,
    PRIMARY KEY (corpus_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_docs_corpus_scope_time
    ON docs(corpus_id, scope_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_docs_corpus_user_time
    ON docs(corpus_id, user_id, created_at_ts DESC);

CREATE INDEX IF NOT EXISTS idx_docs_corpus_org_time
    ON docs(corpus_id, org_id, created_at_ts DESC);
"""


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


@dataclass
class SQLiteLexicalSearchBackend(SearchBackend):
    """
    Cheap non-LLM search backend.

    - Upsert: store raw text + metadata in a SQLite table.
    - Search: use simple keyword LIKE search + identity/time filters.


    Right now the lexical backend is a simple bag-of-words search over a SQLite table:
    - Every upsert stores: corpus_id, item_id, raw text, full meta_json, and promoted fields (org_id, user_id, scope_id, run_id, kind, source, created_at_ts, etc.) into docs.
    - At query time, we:
        - Use SQL to filter by corpus, org/user/scope, and optional time window (created_at_min/max), and sort by created_at_ts DESC LIMIT N (recency bias).
        - Pull that candidate row set into Python.
        - Tokenize the query ("sample text JSON artifact" → ["sample", "text", "json", "artifact"]).
        - For each candidate text, count how many tokens appear (and how often), and derive a simple score:
        - “more distinct query words present + a tiny bump for repeats = higher score.”
        - Discard docs that match none of the tokens, return top-k by score.

    NOTE: it’s exact token match, multi-word aware, and understands time + scope, but deliberately dumb:
    - No stemming (“run” vs “running”), no synonyms, no typo/fuzzy matching.
    - No real IR scoring (no TF-IDF/BM25, no field weighting, no phrase queries).
    - Quality will degrade for huge corpora because ranking is naive and all ranking happens in Python.
    - But it’s cheap, local, deterministic, and good enough for “I remember some words from that thing I saved.”
    """

    db_path: str

    def __post_init__(self) -> None:
        _ensure_db(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # -------- helpers ----------------------------------------------------

    @staticmethod
    def _parse_time_window(
        time_window: str | None,
        created_at_min: float | None,
        created_at_max: float | None,
    ) -> tuple[float | None, float | None]:
        if not time_window:
            return created_at_min, created_at_max

        if created_at_min is not None and created_at_max is not None:
            return created_at_min, created_at_max

        # very simple parser: "7d", "24h", "30m", "60s"
        import re

        m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", time_window)
        if not m:
            return created_at_min, created_at_max

        value = int(m.group(1))
        unit = m.group(2)
        factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]

        now_ts = time.time()
        duration = value * factor

        if created_at_min is None:
            created_at_min = now_ts - duration
        if created_at_max is None:
            created_at_max = now_ts

        return created_at_min, created_at_max

    # -------- public APIs -----------------------------------------------

    async def upsert(
        self,
        *,
        corpus: str,
        item_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Store text + metadata in docs table.

        We mirror common promoted fields into columns for cheap filtering.
        """
        if not text:
            text = ""

        # Extract promoted fields from metadata
        org_id = metadata.get("org_id")
        user_id = metadata.get("user_id")
        scope_id = metadata.get("scope_id")
        client_id = metadata.get("client_id")
        app_id = metadata.get("app_id")
        session_id = metadata.get("session_id")
        run_id = metadata.get("run_id")
        graph_id = metadata.get("graph_id")
        node_id = metadata.get("node_id")
        kind = metadata.get("kind")
        source = metadata.get("source")
        created_at_ts = metadata.get("created_at_ts")

        # If no created_at_ts given, fallback to "now" (cheap and good enough)
        if created_at_ts is None:
            created_at_ts = time.time()

        meta_json = json.dumps(metadata, ensure_ascii=False)
        print(f"🍏 Upserting doc {item_id} into corpus {corpus} with metadata {metadata}")
        print(f"🍏 Text: {text}")

        def _upsert_sync() -> None:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    REPLACE INTO docs(
                        corpus_id,
                        item_id,
                        text,
                        meta_json,
                        created_at_ts,
                        org_id,
                        user_id,
                        scope_id,
                        client_id,
                        app_id,
                        session_id,
                        run_id,
                        graph_id,
                        node_id,
                        kind,
                        source
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        corpus,
                        item_id,
                        text,
                        meta_json,
                        float(created_at_ts),
                        org_id,
                        user_id,
                        scope_id,
                        client_id,
                        app_id,
                        session_id,
                        run_id,
                        graph_id,
                        node_id,
                        kind,
                        source,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_upsert_sync)

    async def search(
        self,
        *,
        corpus: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[ScoredItem]:
        if not query.strip():
            return []

        filters = filters or {}

        # Compute final time bounds
        created_at_min, created_at_max = self._parse_time_window(
            time_window, created_at_min, created_at_max
        )

        # We’ll do a cheap LIKE search on text and apply filters in SQL where possible,
        # remaining filters in Python.

        def _search_sync() -> list[ScoredItem]:
            conn = self._connect()
            try:
                cur = conn.cursor()

                sql = """
                    SELECT item_id, text, meta_json, created_at_ts
                    FROM docs
                    WHERE corpus_id = ?
                """
                params: list[Any] = [corpus]

                # subset of filters we can push into SQL
                promoted_cols = {
                    "org_id",
                    "user_id",
                    "scope_id",
                    "client_id",
                    "app_id",
                    "session_id",
                    "run_id",
                    "graph_id",
                    "node_id",
                    "kind",
                    "source",
                }

                sql_filters: dict[str, Any] = {}
                py_filters: dict[str, Any] = {}

                for k, v in filters.items():
                    if v is None:
                        continue
                    if k in promoted_cols and not isinstance(v, (list, tuple, set)):  # noqa: UP038
                        sql_filters[k] = v
                    else:
                        py_filters[k] = v

                for key, val in sql_filters.items():
                    sql += f" AND {key} = ?"
                    params.append(val)

                # Time window
                if created_at_min is not None:
                    sql += " AND created_at_ts >= ?"
                    params.append(created_at_min)
                if created_at_max is not None:
                    sql += " AND created_at_ts <= ?"
                    params.append(created_at_max)

                # Bias toward recent, like vector backend
                sql += " ORDER BY created_at_ts DESC LIMIT ?"
                params.append(max(top_k * 50, top_k))

                cur.execute(sql, params)
                rows = cur.fetchall()
            finally:
                conn.close()

            # Build results, apply any remaining filters in Python, and
            # assign a simple "score" (e.g., count of occurrences)
            results: list[ScoredItem] = []
            print(f"🍏 Retrieved {len(rows)} candidate rows from DB")
            print(f"🍏 Searching corpus {corpus} for query {query} with filters {py_filters}")

            # Basic bag-of-words: split query into tokens
            tokens = [t for t in query.lower().split() if t]

            for item_id, text, meta_json, _ in rows:
                meta = json.loads(meta_json)

                # Python-level filters (e.g., list-valued filters)
                match = True
                for key, val in py_filters.items():
                    if key not in meta:
                        match = False
                        break
                    mv = meta[key]
                    if not self._match_value(mv, val):
                        match = False
                        break
                if not match:
                    continue

                text_lower = (text or "").lower()

                # Naive scoring: token-based exact matches
                match_tokens = 0
                total_hits = 0
                for tok in tokens:
                    c = text_lower.count(tok)
                    if c > 0:
                        match_tokens += 1
                        total_hits += c

                # If none of the tokens appear, skip
                if match_tokens == 0:
                    continue

                # Score: prioritize docs that match more distinct tokens,
                # with a small bump for repeated occurrences.
                score = float(match_tokens) + 0.1 * float(total_hits)

                results.append(
                    ScoredItem(
                        item_id=item_id,
                        corpus=corpus,
                        score=score,
                        metadata=meta,
                    )
                )

                if len(results) >= top_k:
                    break

            return results

        return await asyncio.to_thread(_search_sync)

    @staticmethod
    def _match_value(mv: Any, val: Any) -> bool:
        """
        Rich matching semantics for filters:
        - If val is list/tuple/set:
            - if mv is list-like too -> match if intersection is non-empty
            - else                    -> match if mv is in val
        - If val is scalar:
            - if mv is list-like      -> match if val is in mv
            - else                    -> match if mv == val
        """
        if val is None:
            return True

        def _is_list_like(x: Any) -> bool:
            return isinstance(x, (list, tuple, set))  # noqa: UP038

        if _is_list_like(val):
            if _is_list_like(mv):
                # any overlap between filter values and meta values
                return any(x in val for x in mv)
            else:
                # meta is scalar, filter is list-like
                return mv in val

        # val is scalar
        if _is_list_like(mv):
            return val in mv

        return mv == val
