from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

EventKind = Literal[
    "user_msg",
    "assistant_msg",
    "tool_start",
    "tool_result",
    "error",
    "checkpoint",
    "run_summary",
    "rolling_summary",
]


@dataclass
class Event:
    event_id: str
    ts: str
    run_id: str
    graph_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    tool: str | None = None  # now used for tool topic: TODO: rename to topic in future
    kind: EventKind = "tool_result"
    stage: str | None = None
    severity: int = 2
    signal: float = 0.0
    tags: list[str] | None = None
    entities: list[str] | None = None
    inputs: list[Value] | None = None
    outputs: list[Value] | None = None
    inputs_ref: dict[str, Any] | None = None
    outputs_ref: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    text: str | None = None
    embedding: list[float] | None = None
    pii_flags: dict[str, bool] | None = None
    sources: list[str] | None = None
    version: int = 1  # for schema evolution


class HotLog(Protocol):
    async def append(self, run_id: str, evt: Event, *, ttl_s: int, limit: int) -> None: ...
    async def recent(
        self, run_id: str, *, kinds: list[str] | None = None, limit: int = 50
    ) -> list[Event]: ...


class Persistence(Protocol):
    async def append_event(self, run_id: str, evt: Event) -> None: ...
    async def save_json(self, uri: str, obj: dict[str, Any]) -> None: ...


class Indices(Protocol):
    async def update(self, run_id: str, evt: Event) -> None: ...
    async def last_by_name(self, run_id: str, name: str) -> dict[str, Any] | None: ...
    async def latest_refs_by_kind(
        self, run_id: str, kind: str, *, limit: int = 50
    ) -> list[dict[str, Any]]: ...
    async def last_outputs_by_topic(self, run_id: str, topic: str) -> dict[str, Any] | None: ...


class Distiller(Protocol):
    async def distill(
        self, run_id: str, *, hotlog: HotLog, persistence: Persistence, indices: Indices, **kw
    ) -> dict[str, Any]: ...


# ---------- Vector Index and Embeddings Client Protocols ----------
class VectorIndex(Protocol):
    async def upsert(self, *, id: str, vector: list[float], metadata: dict) -> None: ...
    async def delete(self, *, id: str) -> None: ...
    async def query(
        self, *, vector: list[float], k: int = 8, filter: dict | None = None
    ) -> list[dict]: ...
    async def flush(self) -> None: ...


class EmbeddingsClient(Protocol):
    async def embed_text(self, text: str, *, model: str | None = None) -> list[float]: ...
    async def embed_texts(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]: ...


# ---------- I/O Value and Ref schemas ----------
class Ref(TypedDict, total=False):
    """A resolvable refernece to an external artifact or data."""

    kind: str  # e.g. "spec", "design", "output", "tool_result"
    uri: str  # e.g. "file://...", "mem://...", "db://..."
    title: str | None  # optional human-readable title
    mime: str | None  # optional MIME type, e.g. "image/png"


class Value(TypedDict, total=False):
    """
    A named I/O slot that can hold any JSON-serializable value, including a Ref.
    vtype declares the JSON type; if vtype == "ref", value must be a Ref dict.
    """

    name: str
    vtype: Literal["ref", "number", "string", "boolean", "object", "array", "null"]
    value: Any  # actual value; type depends on vtype
    meta: dict[str, Any] | None  # optional metadata dictionary
