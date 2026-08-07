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
    "external.resource.changed",
]

EXTERNAL_RESOURCE_CHANGED_KIND = "external.resource.changed"
_EXTERNAL_RESOURCE_CHANGE_FIELDS = frozenset(
    {
        "kind",
        "event_id",
        "scope_id",
        "session_id",
        "source_sequence",
        "resource_key",
        "resource_kind",
        "previous_revision",
        "revision",
        "previous_content_hash",
        "content_hash",
        "changed_fields",
        "summary",
        "source",
        "effective_at",
        "recorded_at",
    }
)


@dataclass(frozen=True)
class ExternalResourceChangedEvent:
    event_id: str
    scope_id: str
    session_id: str
    source_sequence: int
    resource_key: str
    resource_kind: str
    revision: str
    source: str
    recorded_at: str
    previous_revision: str = ""
    previous_content_hash: str = ""
    content_hash: str = ""
    changed_fields: tuple[str, ...] = ()
    summary: str = ""
    effective_at: str = ""

    def __post_init__(self) -> None:
        required = {
            "event_id": self.event_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "resource_key": self.resource_key,
            "resource_kind": self.resource_kind,
            "revision": self.revision,
            "source": self.source,
            "recorded_at": self.recorded_at,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(
                "external.resource.changed missing required fields: " + ", ".join(sorted(missing))
            )
        if ":" not in str(self.resource_key):
            raise ValueError("external resource_key must use a namespaced identity")
        if isinstance(self.source_sequence, bool) or int(self.source_sequence) <= 0:
            raise ValueError("external source_sequence must be a positive integer")
        _require_aware_iso_timestamp(self.recorded_at, field_name="recorded_at")
        if self.effective_at:
            _require_aware_iso_timestamp(self.effective_at, field_name="effective_at")
        normalized_fields = tuple(
            dict.fromkeys(
                str(value or "").strip()
                for value in self.changed_fields
                if str(value or "").strip()
            )
        )
        if len(normalized_fields) > 256 or any(
            len(field_path) > 512 for field_path in normalized_fields
        ):
            raise ValueError("external changed_fields exceed compact event limits")
        if len(str(self.summary or "")) > 2000:
            raise ValueError("external summary exceeds compact event limit")
        object.__setattr__(self, "changed_fields", normalized_fields)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExternalResourceChangedEvent:
        """Validate a committed producer-outbox row as one external change.

        Unknown fields are rejected so the compact event cannot accidentally
        become a carrier for the authoritative configuration or other large
        content. The producer's transaction remains responsible for mutation
        plus outbox insertion.

        Examples:
            Parse a design change:
            ```python
                event = ExternalResourceChangedEvent.from_dict(
                    {
                        "kind": "external.resource.changed",
                        "event_id": "evt-19",
                        "scope_id": "session:s-1",
                        "session_id": "s-1",
                        "source_sequence": 19,
                        "resource_key": "design_config:project-42",
                        "resource_kind": "design_config",
                        "revision": "19",
                        "source": "design_ui",
                        "recorded_at": "2026-07-10T20:00:01Z",
                    }
                )
                assert event.source_sequence == 19
            ```

            Reject an embedded large configuration:
            ```python
                try:
                    ExternalResourceChangedEvent.from_dict({"config": {"large": True}})
                except ValueError as exc:
                    assert "unknown fields" in str(exc)
            ```

        Args:
            value: Mapping read from a committed authoritative-store outbox row.

        Returns:
            ExternalResourceChangedEvent: Strict, compact, scope-addressed event.

        Notes:
            `source_sequence` is monotonic within one source and scope and is the
            durable ordering cursor supplied by the authoritative outbox.
        """

        raw = dict(value or {})
        unknown = sorted(set(raw) - _EXTERNAL_RESOURCE_CHANGE_FIELDS)
        if unknown:
            raise ValueError(
                "external.resource.changed contains unknown fields: " + ", ".join(unknown)
            )
        kind = str(raw.get("kind") or EXTERNAL_RESOURCE_CHANGED_KIND)
        if kind != EXTERNAL_RESOURCE_CHANGED_KIND:
            raise ValueError(
                f"external resource event kind must be {EXTERNAL_RESOURCE_CHANGED_KIND!r}"
            )
        try:
            source_sequence = int(raw.get("source_sequence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("external source_sequence must be a positive integer") from exc
        changed_fields = raw.get("changed_fields") or ()
        if not isinstance(changed_fields, list | tuple):
            raise ValueError("external changed_fields must be a list of field paths")
        return cls(
            event_id=str(raw.get("event_id") or "").strip(),
            scope_id=str(raw.get("scope_id") or "").strip(),
            session_id=str(raw.get("session_id") or "").strip(),
            source_sequence=source_sequence,
            resource_key=str(raw.get("resource_key") or "").strip(),
            resource_kind=str(raw.get("resource_kind") or "").strip(),
            previous_revision=str(raw.get("previous_revision") or "").strip(),
            revision=str(raw.get("revision") or "").strip(),
            previous_content_hash=str(raw.get("previous_content_hash") or "").strip(),
            content_hash=str(raw.get("content_hash") or "").strip(),
            changed_fields=tuple(changed_fields),
            summary=str(raw.get("summary") or "").strip(),
            source=str(raw.get("source") or "").strip(),
            effective_at=str(raw.get("effective_at") or "").strip(),
            recorded_at=str(raw.get("recorded_at") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compact event for memory persistence and ingestion.

        Empty optional metadata is omitted while required identity, scope,
        sequence, revision, and source fields remain explicit.

        Examples:
            Serialize required fields:
            ```python
                event = ExternalResourceChangedEvent(
                    event_id="evt-1",
                    scope_id="session:s-1",
                    session_id="s-1",
                    source_sequence=1,
                    resource_key="clock:world",
                    resource_kind="clock",
                    revision="day-2",
                    source="world_service",
                    recorded_at="2026-07-10T20:00:01Z",
                )
                assert event.to_dict()["kind"] == "external.resource.changed"
            ```

            Preserve ordered changed fields:
            ```python
                event = ExternalResourceChangedEvent(
                    event_id="evt-2",
                    scope_id="session:s-1",
                    session_id="s-1",
                    source_sequence=2,
                    resource_key="design:p-1",
                    resource_kind="design",
                    revision="2",
                    source="ui",
                    recorded_at="2026-07-10T20:00:02Z",
                    changed_fields=("lens.aperture",),
                )
                assert event.to_dict()["changed_fields"] == ["lens.aperture"]
            ```

        Args:
            None.

        Returns:
            dict[str, Any]: JSON-safe committed-outbox event payload.

        Notes:
            The payload intentionally has no field for authoritative resource
            content.
        """

        payload: dict[str, Any] = {
            "kind": EXTERNAL_RESOURCE_CHANGED_KIND,
            "event_id": self.event_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "source_sequence": self.source_sequence,
            "resource_key": self.resource_key,
            "resource_kind": self.resource_kind,
            "revision": self.revision,
            "source": self.source,
            "recorded_at": self.recorded_at,
        }
        for key, value in (
            ("previous_revision", self.previous_revision),
            ("previous_content_hash", self.previous_content_hash),
            ("content_hash", self.content_hash),
            ("summary", self.summary),
            ("effective_at", self.effective_at),
        ):
            if value:
                payload[key] = value
        if self.changed_fields:
            payload["changed_fields"] = list(self.changed_fields)
        return payload


def _require_aware_iso_timestamp(value: str, *, field_name: str) -> None:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"external {field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"external {field_name} must include a timezone")


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

    async def append_event(
        self,
        *,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs=None,
        outputs=None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
    ) -> Event: ...

    async def append_external_resource_change(
        self,
        change: ExternalResourceChangedEvent | dict[str, Any],
    ) -> Event: ...

    async def append_chat_turn(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event: ...

    async def append_tool_result(
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

    async def append_state_snapshot(
        self,
        key: str,
        value: Any,
        *,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
        kind: str = "state.snapshot",
        stage: str | None = None,
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

    async def query_events(
        self,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: str | None = None,
        use_persistence: bool = False,
        since: str | None = None,
        until: str | None = None,
        offset: int = 0,
        return_event: bool = True,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        order_dir: Literal["asc", "desc"] = "desc",
    ) -> list[Any]: ...

    async def get_latest_state(
        self,
        key: str,
        *,
        tags=None,
        level: str | None = None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> Any | None: ...

    async def get_latest_state_record(
        self,
        key: str,
        *,
        tags=None,
        level: str | None = None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> dict[str, Any] | None: ...

    async def list_state_history(
        self,
        key: str,
        *,
        tags=None,
        limit: int = 50,
        level: str | None = None,
        kind: str = "state.snapshot",
        use_persistence: bool = False,
    ) -> list[Event]: ...

    async def search_state(
        self,
        query: str,
        *,
        key: str | None = None,
        tags=None,
        top_k: int = 10,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[Any]: ...

    async def distill_summary(self, **kwargs) -> dict[str, Any]: ...

    async def list_summaries(self, **kwargs) -> list[dict[str, Any]]: ...

    async def get_latest_summary(self, *args, **kwargs) -> dict[str, Any] | None: ...


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
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_dir: Literal["asc", "desc"] = "desc",
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
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_dir: Literal["asc", "desc"] = "desc",
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
