from __future__ import annotations

from typing import Any, Protocol


class LexicalIndex(Protocol):
    """
    Minimal protocol for a lexical (FTS/BM25-style) index.

    We keep the interface intentionally close to VectorIndex.search so
    GenericSearchBackend can orchestrate both.
    """

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]],
    ) -> None: ...

    async def delete(self, corpus_id: str, chunk_ids: list[str] | None = None) -> None: ...

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
        Returns rows shaped like:
          {"chunk_id": str, "score": float, "meta": dict[str, Any]}
        """
        ...
