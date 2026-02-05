from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.storage.search_backend import ScoredItem, SearchBackend


@dataclass
class NullSearchBackend(SearchBackend):
    """A no-op search backend that performs no indexing or searching."""

    async def upsert(
        self,
        *,
        corpus: str,
        item_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        # no-op
        return

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
        # either empty or raise FeatureDisabled
        return []
