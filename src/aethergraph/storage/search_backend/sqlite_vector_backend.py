from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend
from aethergraph.contracts.storage.vector_index import VectorIndex


@dataclass
class SQLiteVectorSearchBackend(SearchBackend):
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
            if isinstance(v, list | tuple | set):
                if mv not in v:
                    return False
            else:
                if mv != v:
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
