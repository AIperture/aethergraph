from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

SearchMode = Literal["auto", "semantic", "lexical", "hybrid", "structural"]


@dataclass
class ScoredItem:
    item_id: str
    corpus: str
    score: float
    metadata: dict[str, Any]


class SearchBackend(Protocol):
    async def upsert(
        self,
        *,
        corpus: str,
        item_id: str,
        text: str,
        metadata: Mapping[str, Any],
    ) -> None: ...

    async def search(
        self,
        *,
        corpus: str,
        query: str,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
        mode: SearchMode = "auto",
    ) -> Sequence[ScoredItem]:
        """
        mode:
          - "auto": legacy behavior
              * empty query   -> structural (recency via vector index)
              * non-empty     -> semantic vector search
          - "structural": structural/recency-only (ignores query content)
          - "semantic": semantic vector search (ANN + filters)
          - "lexical": lexical / FTS search (if backend supports it)
          - "hybrid": semantic + lexical merged
        """
        ...
