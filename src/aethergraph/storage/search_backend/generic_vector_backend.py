from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import time
from typing import Any

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend
from aethergraph.contracts.storage.vector_index import PROMOTED_FIELDS, VectorIndex

from .utils import _parse_time_window


@dataclass
class GenericVectorSearchBackend(SearchBackend):
    """
    SearchBackend implementation on top of a VectorIndex + EmbeddingClient.

    - Upserts: embed text and store (vector, metadata) in the index
    - Search: embed query, retrieve top-k by cosine similarity,
      then apply Python-level metadata filters.
    """

    index: VectorIndex
    embedder: EmbeddingClientProtocol

    # -------- helpers ----------------------------------------------------
    async def _embed(self, text: str) -> list[float]:
        vec: Sequence[float] = await self.embedder.embed_one(text)
        # Ensure a concrete list[float] for numpy/etc
        return [float(x) for x in vec]

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

    @staticmethod
    def _matches_filters(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        """
        Simple AND filter: all filter keys must match exactly.
        - If filter value is a list, meta[key] must be in that list.
        - If filter value is None, we don't constrain that key.
        """
        for k, v in filters.items():
            if v is None:
                continue
            if k not in meta:
                return False
            mv = meta[k]
            if not GenericVectorSearchBackend._match_value(mv, v):
                return False
        return True

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
            # avoid zero vector; caller should ensure text is non-empty
            text = ""

        vector = await self._embed(text)
        await self.index.add(
            corpus_id=corpus,
            chunk_ids=[item_id],
            vectors=[vector],
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
    ) -> list[ScoredItem]:
        filters = filters or {}
        if not query.strip():
            return []

        q_vec = await self._embed(query)

        # ---- 1) Handle time constraints ---------------------------------
        now_ts = time()

        # If time_window is provided and no explicit min, interpret it as [now - window, now]
        if time_window and created_at_min is None:
            duration = _parse_time_window(time_window)
            created_at_min = now_ts - duration

        # If max is not provided but we used a time_window, default to now
        if time_window and created_at_max is None:
            created_at_max = now_ts

        # ---- 2) Split filters into index-level vs Python-level ---------
        index_filters: dict[str, Any] = {}
        post_filters: dict[str, Any] = {}

        for key, val in filters.items():
            if val is None:
                continue

            if key in PROMOTED_FIELDS and not isinstance(val, (list, tuple, set)):  # noqa: UP038
                index_filters[key] = val
            else:
                post_filters[key] = val

        # ---- 3) Ask index for scoped, time-bounded candidates ----------
        raw_k = max(top_k * 3, top_k)
        max_candidates = max(top_k * 50, raw_k)  # tunable safety cap

        rows = await self.index.search(
            corpus_id=corpus,
            query_vec=q_vec,
            k=raw_k,
            where=index_filters,
            max_candidates=max_candidates,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

        # ---- 4) Apply Python-level filters + build ScoredItem list -----
        results: list[ScoredItem] = []
        for row in rows:
            chunk_id = row["chunk_id"]
            score = float(row["score"])
            meta = dict(row.get("meta") or {})

            if post_filters and not self._matches_filters(meta, post_filters):
                continue

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

    async def search_old(
        self,
        *,
        corpus: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredItem]:
        """
        1) Embed the query
        2) Vector search in the underlying index
        3) Apply metadata filters in Python
        """
        filters = filters or {}
        if not query.strip():
            # empty query: probably return nothing for now
            return []

        q_vec = await self._embed(query)

        # Ask underlying VectorIndex for more than top_k, since we may
        # filter some out. Factor 3 is arbitrary but usually safe.
        raw_k = max(top_k * 3, top_k)
        rows = await self.index.search(
            corpus_id=corpus,
            query_vec=q_vec,
            k=raw_k,
        )

        results: list[ScoredItem] = []
        for row in rows:
            chunk_id = row["chunk_id"]
            score = float(row["score"])
            meta = dict(row.get("meta") or {})

            if filters and not self._matches_filters(meta, filters):
                continue

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
