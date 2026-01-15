from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.storage.search_backend import ScoredItem

ContentMode = Literal["json", "text", "bytes"]

ArtifactView = Literal["node", "graph", "run", "all"]


class ArtifactSearchResult(NamedTuple):
    item: ScoredItem  # raw search result (score, metadata, etc.)
    artifact: Artifact | None  # hydrated Artifact, or None if missing

    @property
    def id(self) -> str:
        return self.item.item_id

    @property
    def score(self) -> float:
        return self.item.score

    @property
    def meta(self) -> dict[str, Any]:
        return self.item.metadata


@dataclass
class ArtifactContent:
    artifact: Artifact
    mode: ContentMode  # "json", "text", or "bytes"
    text: str | None = None
    json: Any | None = None
    data: bytes | None = None
