from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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
        metadata: dict[str, Any],
    ) -> None:
        """
        Insert or update an indexed item.

        - corpus: logical collection ("event", "artifact", "run", "doc_*")
        - item_id: stable identifier (event_id, artifact_id, run_id, etc.)
        - text: main text used for embedding / lexical search
        - metadata: arbitrary JSON metadata for filters and recency
        """
        ...

    async def search(
        self,
        *,
        corpus: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredItem]:
        """
        Semantic/lexical search.

        - filters: AND filters over metadata (None values are treated as wildcards).
        """
        ...
