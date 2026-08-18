from dataclasses import dataclass
from typing import Any, Literal

from aethergraph.contracts.services.artifacts import Artifact

ContentMode = Literal["json", "text", "bytes"]


@dataclass
class ArtifactContent:
    artifact: Artifact
    mode: ContentMode  # "json", "text", or "bytes"
    text: str | None = None
    json: Any | None = None
    data: bytes | None = None
