from dataclasses import dataclass
from typing import Literal

from aethergraph.contracts import JsonValue
from aethergraph.contracts.services.artifacts import Artifact

ContentMode = Literal["json", "text", "bytes"]


@dataclass
class ArtifactContent:
    artifact: Artifact
    mode: ContentMode  # "json", "text", or "bytes"
    text: str | None = None
    json: JsonValue = None
    data: bytes | None = None
