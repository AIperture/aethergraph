from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import time
from typing import Any

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.storage.lexical_index import LexicalIndex
from aethergraph.contracts.storage.search_backend import (
    ScoredItem,
    SearchBackend,
    SearchMode,
)
from aethergraph.contracts.storage.vector_index import PROMOTED_FIELDS, VectorIndex

from .utils import _parse_time_window

"""
Search modes:

- auto:
    * If query is empty/whitespace -> structural
    * Else                        -> semantic
    (Preserves legacy behavior.)

- structural:
    * Ignores query content.
    * Uses VectorIndex with empty query_vec, which triggers "no-query" path:
        - Filter by promoted fields + time window.
        - Order by created_at_ts DESC.
        - Score = created_at_ts (recency).
    * Use for "latest items in this scope".

- semantic:
    * Embeds query via EmbeddingClient.
    * VectorIndex ANN search + filters + time window.
    * Score = cosine similarity (or FAISS equivalent).
    * Use for meaning-based memory / doc search.

- lexical:
    * If LexicalIndex is configured:
        - FTS5 MATCH query with bm25() score.
        - Filter by promoted fields + time window.
      Else:
        - Falls back to semantic.
    * Use for exact keyword / code / error-text search.

- hybrid:
    * If LexicalIndex is configured and query is non-empty:
        - Run semantic + lexical.
        - Normalize each channel's scores to [0,1].
        - Combined score = alpha * semantic + (1-alpha) * lexical.
        - Sort by combined score.
      Else:
        - Falls back to semantic.
    * Use for KB / doc QA where both semantics and exact matches matter.
"""


@dataclass
class GenericSearchBackend(SearchBackend):
    """
    SearchBackend implementation on top of:

      - a VectorIndex + EmbeddingClient (semantic + structural),
      - an optional LexicalIndex (FTS/BM25).

    - Upserts: embed text and store (vector, metadata) in the VectorIndex;
      optionally also index text in the LexicalIndex.
    - Search:
        * structural (no query): recency + metadata filters using VectorIndex only
        * semantic: vector ANN + metadata filters
        * lexical: FTS/BM25 + metadata filters
        * hybrid: semantic + lexical merged and re-ranked

    The public `upsert()` and `search()` signatures match SearchBackend with
    an added `mode` parameter.
    """

    index: VectorIndex
    embedder: EmbeddingClientProtocol
    lexical: LexicalIndex | None = None
    debug: bool = True

    # -------- helpers ----------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        vec: Sequence[float] = await self.embedder.embed_one(text)
        return [float(x) for x in vec]

    @staticmethod
    def _match_value(mv: Any, val: Any) -> bool:
        if val is None:
            return True

        def _is_list_like(x: Any) -> bool:
            return isinstance(x, (list, tuple, set))  # noqa: UP038

        if _is_list_like(val):
            if _is_list_like(mv):
                return any(x in val for x in mv)
            else:
                return mv in val

        if _is_list_like(mv):
            return val in mv

        return mv == val

    @staticmethod
    def _matches_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for k, v in filters.items():
            if v is None:
                continue
            if k not in meta:
                return False
            mv = meta[k]
            if not GenericSearchBackend._match_value(mv, v):
                return False
        return True

    @staticmethod
    def _split_filters(filters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Split filters into:
          - index_filters: scalar promoted fields for pushdown into VectorIndex/LexicalIndex
          - post_filters: everything else (tags, list-valued, etc.)
        """
        index_filters: dict[str, Any] = {}
        post_filters: dict[str, Any] = {}

        for key, val in filters.items():
            if val is None:
                continue

            if key in PROMOTED_FIELDS and not isinstance(val, (list, tuple, set)):  # noqa: UP038
                index_filters[key] = val
            else:
                post_filters[key] = val

        return index_filters, post_filters

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        mn = min(vals)
        mx = max(vals)
        if mx - mn < 1e-9:
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

    @staticmethod
    def _merge_hybrid_rows(
        semantic_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        combined_k: int,
        alpha: float = 0.6,
        *,
        debug: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Merge semantic + lexical results into a single ranked list.

        - alpha: weight for semantic score; (1 - alpha) for lexical.
        - combined_k: cap on total items returned (pre post-filtering).

        If debug=True, attach a reserved `_ag_search_debug` block to each row.meta:
            {
              "channels": {"semantic": bool, "lexical": bool},
              "semantic": {"raw": float | None, "norm": float | None},
              "lexical":  {"raw": float | None, "norm": float | None},
              "combined": {"score": float},
            }
        """
        sem_scores = {r["chunk_id"]: float(r["score"]) for r in semantic_rows}
        lex_scores = {r["chunk_id"]: float(r["score"]) for r in lexical_rows}

        sem_norm = GenericSearchBackend._normalize_scores(sem_scores)
        lex_norm = GenericSearchBackend._normalize_scores(lex_scores)

        combined: dict[str, tuple[float, dict[str, Any]]] = {}

        # helper to attach debug info into meta
        def _attach_debug(
            *,
            cid: str,
            meta: dict[str, Any],
            score: float,
        ) -> None:
            if not debug:
                return

            dbg = dict(meta.get("_ag_search_debug") or {})
            dbg_channels = dict(dbg.get("channels") or {})
            # mark channel presence
            dbg_channels["semantic"] = dbg_channels.get("semantic", False) or (cid in sem_scores)
            dbg_channels["lexical"] = dbg_channels.get("lexical", False) or (cid in lex_scores)

            dbg["channels"] = dbg_channels
            dbg["semantic"] = {
                "raw": sem_scores.get(cid),
                "norm": sem_norm.get(cid),
            }
            dbg["lexical"] = {
                "raw": lex_scores.get(cid),
                "norm": lex_norm.get(cid),
            }
            dbg["combined"] = {"score": score}
            meta["_ag_search_debug"] = dbg

        # 1) seed from semantic rows
        for r in semantic_rows:
            cid = r["chunk_id"]
            score = alpha * sem_norm.get(cid, 0.0) + (1.0 - alpha) * lex_norm.get(cid, 0.0)
            meta = dict(r.get("meta") or {})
            _attach_debug(cid=cid, meta=meta, score=score)
            combined[cid] = (score, meta)

        # 2) merge lexical rows (may add new cids or enrich existing meta)
        for r in lexical_rows:
            cid = r["chunk_id"]
            score = alpha * sem_norm.get(cid, 0.0) + (1.0 - alpha) * lex_norm.get(cid, 0.0)
            prev = combined.get(cid)
            if prev is None:
                meta = dict(r.get("meta") or {})
            else:
                _, meta = prev  # reuse meta from semantic side
            _attach_debug(cid=cid, meta=meta, score=score)
            combined[cid] = (score, meta)

        # 3) sort & truncate
        items = sorted(combined.items(), key=lambda kv: kv[1][0], reverse=True)
        out: list[dict[str, Any]] = []
        for cid, (score, meta) in items[:combined_k]:
            out.append({"chunk_id": cid, "score": score, "meta": meta})
        return out

    # -------- public APIs ------------------------------------------------

    async def upsert(
        self,
        *,
        corpus: str,
        item_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        if not text:
            text = ""

        # 1) semantic/structural (vector)
        vector = await self._embed(text)
        await self.index.add(
            corpus_id=corpus,
            chunk_ids=[item_id],
            vectors=[vector],
            metas=[metadata],
        )

        # 2) lexical (optional)
        if self.lexical is not None:
            await self.lexical.add(
                corpus_id=corpus,
                chunk_ids=[item_id],
                texts=[text],
                metas=[metadata],
            )

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
        mode: SearchMode = "auto",
    ) -> list[ScoredItem]:
        raw_filters = dict(filters or {})
        index_filters, post_filters = self._split_filters(raw_filters)

        no_query = query is None or not str(query).strip()

        # ---- time constraints -------------------------------------------
        now_ts = time()

        if time_window and created_at_min is None:
            duration = _parse_time_window(time_window)
            created_at_min = now_ts - duration

        if time_window and created_at_max is None:
            created_at_max = now_ts

        # ---- effective mode ---------------------------------------------
        if mode == "auto":
            effective_mode: SearchMode
            effective_mode = "structural" if no_query else "semantic"
        else:
            effective_mode = mode

        # ---- helpers ----------------------------------------------------
        raw_k = max(top_k * 3, top_k)
        max_candidates = max(top_k * 50, raw_k)

        async def _vector_rows(q_text: str | None) -> list[dict[str, Any]]:
            if q_text is None or not q_text.strip():
                q_vec: list[float] = []
            else:
                q_vec = await self._embed(q_text)

            return await self.index.search(
                corpus_id=corpus,
                query_vec=q_vec,
                k=raw_k,
                where=index_filters,
                max_candidates=max_candidates,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )

        async def _lexical_rows() -> list[dict[str, Any]]:
            if self.lexical is None or no_query:
                return []
            out = await self.lexical.search(
                corpus_id=corpus,
                query=query,
                k=raw_k,
                index_filters=index_filters,
                created_at_min=created_at_min,
                created_at_max=created_at_max,
            )
            return out

        # ---- choose path ------------------------------------------------
        if effective_mode == "structural":
            rows = await _vector_rows(None)

        elif effective_mode == "semantic":
            rows = await _vector_rows(query)

        elif effective_mode == "lexical":
            if self.lexical is None:
                rows = await _vector_rows(query)
            else:
                rows = await _lexical_rows()

        elif effective_mode == "hybrid":
            if self.lexical is None or no_query:
                rows = await _vector_rows(query)
            else:
                sem_rows = await _vector_rows(query)
                lex_rows = await _lexical_rows()
                if self.debug:
                    print(
                        f"hybrid search: sem_rows={len(sem_rows)} "
                        f"lex_rows={len(lex_rows)} "
                        f"corpus={corpus!r}"
                    )
                rows = self._merge_hybrid_rows(
                    semantic_rows=sem_rows,
                    lexical_rows=lex_rows,
                    combined_k=raw_k,
                    alpha=0.6,
                    debug=self.debug,
                )
        else:
            # fallback to legacy behavior
            rows = await _vector_rows(query)

        # ---- post-filters + ScoredItem ---------------------------------
        results: list[ScoredItem] = []
        for row in rows:
            chunk_id = row["chunk_id"]
            score = float(row["score"])
            meta = dict(row.get("meta") or {})

            if post_filters and not self._matches_filters(meta, post_filters):
                continue

            if self.debug:
                dbg = dict(meta.get("_ag_search_debug") or {})
                dbg["mode"] = effective_mode
                meta["_ag_search_debug"] = dbg

            results.append(
                ScoredItem(
                    item_id=chunk_id,
                    corpus=corpus,
                    score=score,
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results
