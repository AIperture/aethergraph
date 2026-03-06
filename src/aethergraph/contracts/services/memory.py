from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    """
    A structured event log entry stored in memory.
    This dataclass represents a single event in the system's event log, capturing
    execution context, semantic information, and optional metadata about the event.

    Attributes:
        event_id (str): Unique identifier for this event.
        ts (str): Timestamp when the event occurred.
        run_id (str): Identifier for the execution run containing this event.
        scope_id (str): Identifier for the execution scope.
        user_id (str | None): Optional identifier for the user associated with the event.
        org_id (str | None): Optional identifier for the organization.
        client_id (str | None): Optional identifier for the client.
        session_id (str | None): Optional identifier for the session.
        kind (EventKind): Logical type of the event (e.g., "chat_user", "tool_start").
        stage (str | None): Optional phase indicator (e.g., "user", "assistant", "system", "tool").
        text (str | None): Primary human-readable content of the event (short, may be truncated).
        tags (list[str] | None): Low-cardinality labels for filtering and searching.
        data (dict[str, Any] | None): Arbitrary JSON payload containing event-specific data.
        metrics (dict[str, float] | None): Numeric metrics associated with the event.
        graph_id (str | None): Optional identifier for the graph context.
        node_id (str | None): Optional identifier for the node context.
        tool (str | None): Tool topic associated with the event. Deprecated: use topic instead.
        topic (str | None): Topic classification for the event.
        severity (int): Severity level of the event (1=low, 2=medium, 3=high). Defaults to 2.
        signal (float): Signal strength indicating estimated importance or relevance. Defaults to 0.0.
        inputs (list[Value] | None): Optional input values associated with the event.
        outputs (list[Value] | None): Optional output values associated with the event.
        app_id (str | None): Reserved for schema compatibility.
        agent_id (str | None): Reserved for schema compatibility.
        embedding (list[float] | None): Reserved for future vector payload usage.
        pii_flags (dict[str, bool] | None): Reserved for future PII marker usage.
        version (int): Schema version for tracking schema evolution. Defaults to 2.
    """

    # --------- Core fields ---------
    event_id: str
    ts: str

    # --------- Execution / Tenant Identity ---------
    run_id: str
    scope_id: str
    user_id: str | None = None
    org_id: str | None = None
    client_id: str | None = None
    session_id: str | None = None

    # --------- Core semantics ---------
    kind: EventKind = None  # logical type: "chat_user", "tool_start", etc.
    stage: str | None = None  # optional phase (user/assistant/system/tool, etc.)
    text: str | None = None  # primary human-readable content (short, truncated)
    tags: list[str] | None = None  # low-cardinality labels for filtering/searching
    data: dict[str, Any] | None = None  # arbitrary JSON payload for event-specific data
    metrics: dict[str, float] | None = None  # numeric metrics associated with event

    # --------- Node context ---------
    graph_id: str | None = None
    node_id: str | None = None

    # --------- Optional fields ---------
    tool: str | None = None  # now used for tool topic: TODO: rename to topic in future
    topic: str | None = None
    severity: int = 2  # 1=low, 2=medium, 3=high
    signal: float = 0.0  # signal strength of the event (estimated importance or relevance)
    inputs: list[Value] | None = None  # optional I/O values of the event
    outputs: list[Value] | None = None  # optional I/O values of the event

    # --------- Reserved / seldom-used fields (kept for schema compatibility) ---------
    app_id: str | None = None
    agent_id: str | None = None
    embedding: list[float] | None = None  # reserved for future vector payload usage
    pii_flags: dict[str, bool] | None = None  # reserved for future pii marker usage

    # --------- Schema versioning ---------
    version: int = 2  # for schema evolution


class MemoryTenantFilter(TypedDict, total=False):
    org_id: str
    user_id: str
    client_id: str


class MemoryFacadeProtocol(Protocol):
    """
    Structural protocol for MemoryFacade mixins.

    Mixins type-hint against this protocol instead of a local facade-only type so
    shared contracts live under `contracts.services`.
    """

    run_id: str
    timeline_id: str
    memory_scope_id: str

    hotlog: HotLog
    persistence: Persistence
    scope: Any
    scoped_indices: Any
    llm: Any
    logger: Any

    hot_limit: int
    hot_ttl_s: int
    default_signal_threshold: float

    async def record_raw(
        self,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Event: ...

    async def record(
        self,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs_ref=None,
        outputs_ref=None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,
    ) -> Event: ...

    async def recent(
        self,
        *,
        kinds: list[str] | None = None,
        limit: int = 50,
        level: str | None = None,
        return_event: bool = True,
    ) -> list[Any]: ...

    async def recent_events(
        self,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: str | None = None,
        use_persistence: bool = False,
        return_event: bool = True,
    ) -> list[Any]: ...

    async def record_tool_result(
        self,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event: ...

    async def recent_tool_results(
        self,
        *,
        tool: str,
        limit: int = 10,
        return_event: bool = True,
    ) -> list[Any]: ...


class HotLog(Protocol):
    async def append(self, timeline_id: str, evt: Event, *, ttl_s: int, limit: int) -> None: ...
    async def recent(
        self, timeline_id: str, *, kinds: list[str] | None = None, limit: int = 50
    ) -> list[Event]: ...
    async def query(
        self,
        timeline_id: str,
        *,
        tenant: MemoryTenantFilter | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Event]: ...


class Persistence(Protocol):
    async def append_event(self, timeline_id: str, evt: Event) -> None: ...
    async def save_json(self, uri: str, obj: dict[str, Any]) -> str: ...
    async def load_json(self, uri: str) -> dict[str, Any]: ...
    async def get_events_by_ids(
        self,
        timeline_id: str,
        event_ids: list[str],
        tenant: MemoryTenantFilter | None = None,
    ) -> list[Event]: ...
    async def query_events(
        self,
        timeline_id: str,
        *,
        tenant: MemoryTenantFilter | None = None,
        since: str | None = None,
        until: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Event]: ...
    async def query_summaries(
        self,
        *,
        scope_id: str | None = None,
        timeline_id: str | None = None,
        tenant: MemoryTenantFilter | None = None,
        summary_tag: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...


class Distiller(Protocol):  # or base class
    async def summarize(
        self,
        *,
        events: list[Event],
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
