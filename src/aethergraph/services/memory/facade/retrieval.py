from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from aethergraph.contracts.storage.search_backend import ScoredItem

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event


class EventSearchResult(NamedTuple):
    item: ScoredItem
    event: Event | None

    @property
    def score(self) -> float:
        return self.item.score
