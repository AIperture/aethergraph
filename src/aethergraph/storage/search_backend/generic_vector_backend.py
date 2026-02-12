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

    - Upserts: embed text and store (vector, metadata) in the index.
    - Search: embed query, retrieve top-k by cosine similarity,
      then apply Python-level metadata filters.

    DESIGN NOTES
    ------------
    - Only *promoted* fields (see PROMOTED_FIELDS) are pushed down into the VectorIndex.
      Everything else (e.g. list-valued filters, tags) is handled as a post-filter.
    - Tags:
        For now, tags are intentionally *not* promoted. They should live under
        meta["tags"] (usually a list[str]) and are filtered using _match_value().
        This makes tag behavior consistent across different storage backends.

        If later introduce a dedicated tag index, this is the main place to
        adjust the filter splitting logic:
          * Move "tags" (or some subset) from post_filters into index_filters.
          * Optionally add a separate tag lookup to pre-restrict candidate IDs.
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

        This supports patterns like:
            meta["tags"] = ["tool_result", "surrogate"]
            filter["tags"] = ["surrogate"]
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
        Simple AND filter: all filter keys must match exactly (via _match_value).

        - If filter value is a list, meta[key] must overlap with that list.
        - If filter value is None, we don't constrain that key.

        NOTE:
            This is where tag filters (meta["tags"]) are evaluated today.
            If later add a tag index / tag table:
              - Split tags out earlier (in search()).
              - Use that index to pre-restrict candidate IDs.
              - Keep this as a secondary sanity check.
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
            # Avoid zero vector; caller should ensure text is meaningful.
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
        no_query = query is None or not query.strip()
        # ---- 1) Embed query if present ----------------------------------
        if no_query:
            q_vec: list[float] = []
        else:
            q_vec = await self._embed(query)

        # ---- 2) Handle time constraints --------------------------------
        now_ts = time()

        if time_window and created_at_min is None:
            duration = _parse_time_window(time_window)
            created_at_min = now_ts - duration

        if time_window and created_at_max is None:
            created_at_max = now_ts

        # ---- 3) Split filters into index-level vs Python-level ----------
        index_filters: dict[str, Any] = {}
        post_filters: dict[str, Any] = {}

        for key, val in filters.items():
            if val is None:
                continue

            if key in PROMOTED_FIELDS and not isinstance(val, (list, tuple, set)):  # noqa: UP038
                index_filters[key] = val
            else:
                post_filters[key] = val

        # ---- 4) Ask index for candidates (semantic or structural) ------
        raw_k = max(top_k * 3, top_k)
        max_candidates = max(top_k * 50, raw_k)  # tunable safety cap

        rows = await self.index.search(
            corpus_id=corpus,
            query_vec=q_vec,  # empty => structural search, see VectorIndex contract
            k=raw_k,
            where=index_filters,
            max_candidates=max_candidates,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )

        # ---- 5) Apply Python-level filters + build ScoredItem list -----
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
