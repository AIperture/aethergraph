"""Stable public Memory event behavior over canonical provider records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import json
from typing import Any, Literal
from uuid import uuid4

from aethergraph.contracts.services.memory import Event
from aethergraph.storage.contracts import (
    EventQuery,
    EventRecord,
    PageRequest,
    SortDirection,
)

from .canonical_facade import CanonicalMemoryFacade

_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_MAX_QUERY_EVENTS = 10_000
_PAGE_SIZE = 500


class CanonicalPublicMemoryFacade:
    """Project stable public Memory events onto one canonical facade."""

    def __init__(
        self,
        *,
        canonical: CanonicalMemoryFacade,
        logical_scope_id: str,
        deprecated_app_id: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        event_id_factory: Callable[[], str] = lambda: f"event-{uuid4().hex}",
    ) -> None:
        """Bind public Memory behavior to one exact canonical execution scope.

        The facade translates public Event DTO fields once and delegates persistence
        only to the supplied canonical facade. Construction performs no I/O.

        Examples:
            Bind session Memory:
                ```python
                memory = CanonicalPublicMemoryFacade(
                    canonical=canonical,
                    logical_scope_id="session:session-1",
                )
                ```

            Retain deprecated App response metadata:
                ```python
                memory = CanonicalPublicMemoryFacade(
                    canonical=canonical,
                    logical_scope_id="run:run-1",
                    deprecated_app_id="app-1",
                    clock=clock.now,
                    event_id_factory=lambda: "event-1",
                )
                ```

        Args:
            canonical: Exact canonical event/state/search facade for this scope.
            logical_scope_id: Stable public memory-bucket label; never provider scope.
            deprecated_app_id: Optional explicitly deprecated compatibility metadata.
            clock: Timezone-aware UTC event timestamp source.
            event_id_factory: Stable non-empty event identity source.

        Returns:
            None: The public projection is ready without persistence I/O.

        Notes:
            `logical_scope_id` and deprecated App metadata never affect provider
            selection, authorization, partitioning, or canonical `StorageScope`.
        """
        if not isinstance(logical_scope_id, str) or not logical_scope_id.strip():
            raise ValueError("logical_scope_id must be a non-empty string")
        if logical_scope_id != logical_scope_id.strip():
            raise ValueError("logical_scope_id must not contain surrounding whitespace")
        if deprecated_app_id is not None and (
            not isinstance(deprecated_app_id, str)
            or not deprecated_app_id.strip()
            or deprecated_app_id != deprecated_app_id.strip()
        ):
            raise ValueError("deprecated_app_id must be a non-empty exact string when supplied")
        self.canonical = canonical
        self.scope = canonical.scope
        self.memory_scope_id = logical_scope_id
        self.timeline_id = logical_scope_id
        self._deprecated_app_id = deprecated_app_id
        self._clock = clock
        self._event_id_factory = event_id_factory

    async def append_event(
        self,
        *,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs: list[Any] | None = None,
        outputs: list[Any] | None = None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
    ) -> Event:
        """Append one public Memory event through canonical event persistence.

        Public aliases are normalized before one authoritative append. Search
        projection follows the canonical facade's explicit commit semantics.

        Examples:
            Append a checkpoint:
                ```python
                event = await memory.append_event(
                    kind="checkpoint",
                    data={"step": 2},
                    text="checkpoint two",
                )
                ```

            Append a Tool result with the deprecated alias:
                ```python
                event = await memory.append_event(
                    kind="tool_result",
                    data={"count": 3},
                    tool="search",
                    tags=["verified"],
                )
                ```

        Args:
            kind: Exact non-empty event kind.
            data: JSON-compatible event-specific payload.
            tags: Optional unique non-empty indexed tags.
            severity: Canonical severity from zero through 100.
            stage: Optional exact execution stage.
            inputs: Optional JSON-compatible public input values.
            outputs: Optional JSON-compatible public output values.
            metrics: Optional finite numeric metrics.
            signal: Optional finite relevance signal.
            text: Optional searchable text; compact JSON is derived when absent.
            topic: Optional exact canonical topic.
            tool: Deprecated topic alias; conflicts with `topic` fail directly.

        Returns:
            Event: Persisted public Event DTO reconstructed from the committed record.

        Notes:
            Deprecated `app_id` is retained only in marked payload compatibility
            metadata and never enters canonical scope or indexes.
        """
        resolved_topic = _topic(topic=topic, tool=tool)
        payload: dict[str, Any] = {"data": data}
        if inputs is not None:
            payload["inputs"] = inputs
        if outputs is not None:
            payload["outputs"] = outputs
        if self._deprecated_app_id is not None:
            payload[_COMPATIBILITY_METADATA] = {
                _DEPRECATED_APP_ID: {
                    "value": self._deprecated_app_id,
                    "deprecated": True,
                    "scheduled_removal": "future breaking release",
                }
            }
        occurred_at = _utc(self._clock())
        event_id = _identity(self._event_id_factory())
        searchable_text = text if text is not None else _text(data)
        receipt = await self.canonical.append_event(
            event_id=event_id,
            occurred_at=occurred_at,
            kind=kind,
            stage=stage,
            topic=resolved_topic,
            text=searchable_text,
            tags=_tags(tags),
            payload=payload,
            metrics=dict(metrics or {}),
            severity=severity,
            signal=signal,
        )
        return _public_event(receipt.events[0], logical_scope_id=self.memory_scope_id)

    async def append_chat_turn(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """Append one public chat turn through canonical Memory events.

        Role is promoted to stage while the authored payload and searchable text are
        retained in one canonical event.

        Examples:
            Append user input:
                ```python
                event = await memory.append_chat_turn("user", "Hello")
                ```

            Append an annotated assistant reply:
                ```python
                event = await memory.append_chat_turn(
                    "assistant",
                    "Done",
                    tags=["final"],
                    data={"model": "default"},
                )
                ```

        Args:
            role: Exact supported chat role.
            text: Authored human-readable chat content.
            tags: Optional unique indexed tags.
            data: Optional JSON-compatible chat metadata.
            severity: Canonical severity from zero through 100.
            signal: Optional finite relevance signal.

        Returns:
            Event: Persisted public chat Event DTO.

        Notes:
            Channel emission remains separate; this method only records Memory.
        """
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        return await self.append_event(
            kind="chat.turn",
            stage=role,
            text=text,
            data={"role": role, **dict(data or {})},
            tags=list(_tags(["chat", *(tags or [])])),
            severity=severity,
            signal=signal,
        )

    async def get_event(self, event_id: str) -> Event | None:
        """Read one public Memory event by stable identity.

        The exact canonical scope constrains hydration and a durable miss remains a
        miss without consulting cache or search.

        Examples:
            Hydrate an event:
                ```python
                event = await memory.get_event("event-1")
                ```

            Detect absence:
                ```python
                assert await memory.get_event("missing") is None
                ```

        Args:
            event_id: Exact stable caller-owned event identity.

        Returns:
            Event | None: Public Event DTO or `None` when absent.

        Notes:
            Provider cursors are never accepted as event identifiers.
        """
        record = await self.canonical.get_event(event_id)
        return None if record is None else _public_event(record, self.memory_scope_id)

    async def query_events(
        self,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
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
    ) -> list[Any]:
        """Read bounded public Memory events with canonical opaque paging.

        Durable reads apply canonical indexed filters before paging. Hot reads remain
        explicit and never fall back to persistence after cache eviction.

        Examples:
            Read recent chat from hot memory:
                ```python
                events = await memory.query_events(kinds=["chat.turn"], limit=20)
                ```

            Read durable events after an offset:
                ```python
                rows = await memory.query_events(
                    use_persistence=True,
                    tags=["verified"],
                    offset=10,
                    limit=10,
                    return_event=False,
                    order_dir="asc",
                )
                ```

        Args:
            kinds: Optional exact event kinds.
            tags: Optional tags every event must contain.
            limit: Positive result bound.
            use_persistence: Select durable canonical events instead of hot cache.
            since: Optional inclusive timezone-aware ISO lower time bound.
            until: Optional inclusive timezone-aware ISO upper time bound.
            offset: Bounded public compatibility offset applied over opaque pages.
            return_event: Return Event DTOs when true, normalized mappings otherwise.
            session_id: Optional exact bound-scope session assertion.
            run_id: Optional exact bound-scope run assertion.
            agent_id: Optional exact bound-scope Agent assertion.
            client_id: Deprecated unsupported identity filter; only `None` is accepted.
            graph_id: Optional exact bound-scope graph assertion.
            node_id: Optional exact bound-scope node assertion.
            topic: Optional exact canonical topic filter.
            tool: Deprecated topic alias; conflicts with `topic` fail directly.
            order_dir: Exact ascending or descending provider order.

        Returns:
            list[Any]: Matching Event DTOs or normalized public mappings.

        Notes:
            Offset compatibility is implemented by bounded opaque-cursor traversal;
            integer provider row IDs and unbounded reads are not used.
        """
        _query_bound(limit=limit, offset=offset)
        if order_dir not in {"asc", "desc"}:
            raise ValueError("order_dir must be 'asc' or 'desc'")
        if client_id is not None:
            raise ValueError("client_id is deprecated and is not a canonical Memory filter")
        _assert_scope(
            self.canonical,
            session_id=session_id,
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            agent_id=agent_id,
        )
        resolved_topic = _topic(topic=topic, tool=tool)
        parsed_since = _parse_time(since)
        parsed_until = _parse_time(until)
        if parsed_since is not None and parsed_until is not None and parsed_since > parsed_until:
            raise ValueError("since must not be after until")
        if not use_persistence:
            hot = await self.canonical.recent_hot(
                limit=_MAX_QUERY_EVENTS,
                kinds=tuple(kinds or ()),
                tags=tuple(tags or ()),
            )
            records = [
                record
                for record in hot
                if _record_matches(
                    record,
                    since=parsed_since,
                    until=parsed_until,
                    topic=resolved_topic,
                )
            ]
            if order_dir == "asc":
                records.reverse()
            records = records[offset : offset + limit]
        else:
            records = await self._durable_records(
                kinds=tuple(kinds or ()),
                tags=tuple(tags or ()),
                limit=limit,
                offset=offset,
                since=parsed_since,
                until=parsed_until,
                topic=resolved_topic,
                order=SortDirection.ASCENDING if order_dir == "asc" else SortDirection.DESCENDING,
            )
        events = [_public_event(record, self.memory_scope_id) for record in records]
        return events if return_event else [self.event_to_dict(event) for event in events]

    async def recent_events(
        self,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        use_persistence: bool = False,
        return_event: bool = True,
    ) -> list[Any]:
        """Read bounded newest-first public Memory events.

        This stable convenience method delegates to the same explicit hot-or-durable
        query path and performs no fallback after a miss.

        Examples:
            Read recent hot events:
                ```python
                events = await memory.recent_events(limit=20)
                ```

            Read recent durable chat mappings:
                ```python
                rows = await memory.recent_events(
                    kinds=["chat.turn"],
                    use_persistence=True,
                    return_event=False,
                )
                ```

        Args:
            kinds: Optional exact event kinds.
            tags: Optional tags every event must contain.
            limit: Positive result bound.
            use_persistence: Select durable canonical events instead of hot cache.
            return_event: Return Event DTOs when true, mappings otherwise.

        Returns:
            list[Any]: Matching newest-first Event DTOs or mappings.

        Notes:
            The hot and durable paths are explicit caller choices, not fallback tiers.
        """
        return await self.query_events(
            kinds=kinds,
            tags=tags,
            limit=limit,
            use_persistence=use_persistence,
            return_event=return_event,
            order_dir="desc",
        )

    def event_to_dict(self, event: Event) -> dict[str, Any]:
        """Normalize one public Event DTO to the stable mapping surface.

        The mapping retains compatibility fields for callers while provider cursor
        and private storage metadata remain absent.

        Examples:
            Normalize a chat event:
                ```python
                row = memory.event_to_dict(chat_event)
                ```

            Serialize a result:
                ```python
                payload = json.dumps(memory.event_to_dict(event))
                ```

        Args:
            event: Public Event DTO returned by this facade.

        Returns:
            dict[str, Any]: Detached JSON-compatible public event mapping.

        Notes:
            Optional deprecated `app_id` is response compatibility metadata only.
        """
        return {
            "event_id": event.event_id,
            "ts": event.ts,
            "kind": event.kind,
            "stage": event.stage,
            "text": event.text,
            "tags": list(event.tags or ()),
            "data": event.data,
            "metrics": event.metrics,
            "tool": event.tool,
            "topic": event.topic,
            "severity": event.severity,
            "signal": event.signal,
            "inputs": event.inputs,
            "outputs": event.outputs,
            "run_id": event.run_id,
            "scope_id": event.scope_id,
            "timeline_id": self.timeline_id,
            "session_id": event.session_id,
            "graph_id": event.graph_id,
            "node_id": event.node_id,
            "app_id": event.app_id,
            "agent_id": event.agent_id,
            "user_id": event.user_id,
            "org_id": event.org_id,
            "client_id": event.client_id,
        }

    async def _durable_records(
        self,
        *,
        kinds: tuple[str, ...],
        tags: tuple[str, ...],
        limit: int,
        offset: int,
        since: datetime | None,
        until: datetime | None,
        topic: str | None,
        order: SortDirection,
    ) -> list[EventRecord]:
        selected: list[EventRecord] = []
        cursor: str | None = None
        needed = limit + offset
        while len(selected) < needed:
            page = await self.canonical.durable_query(
                EventQuery(
                    scope=self.scope,
                    page=PageRequest(limit=min(_PAGE_SIZE, needed - len(selected)), cursor=cursor),
                    kinds=kinds,
                    topic=topic,
                    tags=tags,
                    occurred_at_min=since,
                    occurred_at_max=until,
                    order=order,
                )
            )
            selected.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return selected[offset : offset + limit]


def _public_event(record: EventRecord, logical_scope_id: str) -> Event:
    payload = _plain(record.payload)
    compatibility = payload.pop(_COMPATIBILITY_METADATA, {})
    app_id = _deprecated_app_id(compatibility)
    data = payload.pop("data", None)
    inputs = payload.pop("inputs", None)
    outputs = payload.pop("outputs", None)
    if payload:
        if isinstance(data, Mapping):
            data = {**dict(data), "_canonical": payload}
        elif data is None:
            data = payload
    return Event(
        event_id=record.event_id,
        ts=record.occurred_at.isoformat(),
        run_id=record.scope.run_id or "",
        scope_id=logical_scope_id,
        user_id=record.scope.user_id,
        org_id=record.scope.org_id,
        session_id=record.scope.session_id,
        kind=record.kind,
        stage=record.stage,
        text=record.text,
        tags=list(record.tags),
        data=data,
        metrics=dict(record.metrics),
        graph_id=record.scope.graph_id,
        node_id=record.scope.node_id,
        app_id=app_id,
        agent_id=record.scope.agent_id,
        tool=record.topic,
        topic=record.topic,
        severity=record.severity if record.severity is not None else 2,
        signal=record.signal if record.signal is not None else 0.0,
        inputs=inputs if isinstance(inputs, list) else None,
        outputs=outputs if isinstance(outputs, list) else None,
        version=record.schema_version,
    )


def _deprecated_app_id(compatibility: object) -> str | None:
    if compatibility in ({}, None):
        return None
    if not isinstance(compatibility, Mapping) or set(compatibility) != {_DEPRECATED_APP_ID}:
        raise ValueError("Malformed Memory compatibility metadata")
    app = compatibility[_DEPRECATED_APP_ID]
    if (
        not isinstance(app, Mapping)
        or set(app) != {"value", "deprecated", "scheduled_removal"}
        or app.get("deprecated") is not True
        or app.get("scheduled_removal") != "future breaking release"
    ):
        raise ValueError("Malformed deprecated Memory app_id metadata")
    value = app.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Malformed deprecated Memory app_id value")
    return value


def _topic(*, topic: str | None, tool: str | None) -> str | None:
    if topic is not None and tool is not None and topic != tool:
        raise ValueError("tool and topic must match when both are supplied")
    value = topic if topic is not None else tool
    if value is not None and (
        not isinstance(value, str) or not value.strip() or value != value.strip()
    ):
        raise ValueError("topic must be a non-empty string when supplied")
    return value


def _text(data: Any) -> str | None:
    if data is None:
        return None
    if isinstance(data, str):
        return data
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)[:6_000]


def _tags(values: list[str] | None) -> tuple[str, ...]:
    tags = tuple(dict.fromkeys(values or ()))
    if any(not isinstance(tag, str) or not tag.strip() or tag != tag.strip() for tag in tags):
        raise ValueError("Memory tags must contain exact non-empty strings")
    return tags


def _utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError("Memory clock must return a timezone-aware UTC datetime")
    return value


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("event_id_factory must return a non-empty exact string")
    return value


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Memory query timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _query_bound(*, limit: int, offset: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if limit + offset > _MAX_QUERY_EVENTS:
        raise ValueError(f"Memory query exceeds {_MAX_QUERY_EVENTS} records")


def _assert_scope(canonical: CanonicalMemoryFacade, **dimensions: str | None) -> None:
    for name, value in dimensions.items():
        if value is not None and getattr(canonical.scope, name) != value:
            raise ValueError(f"Memory query {name} conflicts with the bound canonical scope")


def _record_matches(
    record: EventRecord,
    *,
    since: datetime | None,
    until: datetime | None,
    topic: str | None,
) -> bool:
    return (
        (since is None or record.occurred_at >= since)
        and (until is None or record.occurred_at <= until)
        and (topic is None or record.topic == topic)
    )


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
