"""Stable public Memory event behavior over canonical provider records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from datetime import UTC, datetime
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from aethergraph.contracts.services.memory import Event
from aethergraph.contracts.storage.event_log import StateSnapshotConflictError
from aethergraph.storage.contracts import (
    EventQuery,
    EventRecord,
    PageRequest,
    SortDirection,
    StateRecord,
    StorageConflictError,
)

from .canonical_facade import CanonicalMemoryFacade
from .canonical_prompt import CanonicalPromptMemoryMixin

if TYPE_CHECKING:
    from aethergraph.contracts.services.llm import LLMClientProtocol

_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_MAX_QUERY_EVENTS = 10_000
_PAGE_SIZE = 500
_MAX_STATE_CAS_RETRIES = 8
_STATE_SERVICE_CONTEXT = "service_context"
_STATE_PUBLIC_METADATA = "public_metadata"


class CanonicalPublicMemoryFacade(CanonicalPromptMemoryMixin):
    """Project stable public Memory events onto one canonical facade."""

    def __init__(
        self,
        *,
        canonical: CanonicalMemoryFacade,
        logical_scope_id: str,
        deprecated_app_id: str | None = None,
        llm: LLMClientProtocol | None = None,
        default_signal_threshold: float = 0.0,
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
                    llm=llm,
                    default_signal_threshold=0.25,
                    clock=clock.now,
                    event_id_factory=lambda: "event-1",
                )
                ```

        Args:
            canonical: Exact canonical event/state/search facade for this scope.
            logical_scope_id: Stable public memory-bucket label; never provider scope.
            deprecated_app_id: Optional explicitly deprecated compatibility metadata.
            llm: Optional explicitly injected LLM client for requested distillation.
            default_signal_threshold: Finite default distillation signal threshold.
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
        if (
            isinstance(default_signal_threshold, bool)
            or not isinstance(default_signal_threshold, int | float)
            or not math.isfinite(default_signal_threshold)
        ):
            raise ValueError("default_signal_threshold must be a finite number")
        self.canonical = canonical
        self.scope = canonical.scope
        self.memory_scope_id = logical_scope_id
        self.timeline_id = logical_scope_id
        self._deprecated_app_id = deprecated_app_id
        self.llm = llm
        self.default_signal_threshold = float(default_signal_threshold)
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
        chat_tags = _tags(tags)
        return await self.append_event(
            kind="chat.turn",
            stage=role,
            text=text,
            data={"role": role, **dict(data or {})},
            tags=list(_tags(["chat", *chat_tags])),
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
        expected_revision: int | None = None,
    ) -> Event:
        """Commit one public state snapshot through canonical state CAS.

        The provider atomically writes current state, history, and its audit/outbox
        row. A public Event DTO is reconstructed from that committed state record
        without duplicating the snapshot into `EventStore`.

        Examples:
            Commit the next state revision:
                ```python
                event = await memory.append_state_snapshot(
                    "agent:writer",
                    {"draft": 2},
                )
                ```

            Compare and set revision three:
                ```python
                event = await memory.append_state_snapshot(
                    "agent:writer",
                    state,
                    expected_revision=2,
                    meta={"reason": "tool_complete"},
                )
                ```

        Args:
            key: Exact caller-owned state key.
            value: JSON-compatible, dataclass, or model state value.
            tags: Optional unique public snapshot tags.
            meta: Optional JSON-compatible public snapshot metadata.
            severity: Canonical severity from zero through 100.
            signal: Optional finite relevance signal.
            kind: Exact state family used for canonical namespace isolation.
            stage: Optional exact execution stage.
            expected_revision: Optional exact current provider revision.

        Returns:
            Event: Public state Event DTO reconstructed from the committed state row.

        Notes:
            Expected conflicts raise `StateSnapshotConflictError`. Unconditional
            writes retry bounded CAS contention on the same store; no alternate store,
            Event append, legacy read, or dual write occurs.
        """
        serialized = _serializable(value)
        public_meta = dict(meta or {})
        normalized_tags = _tags(tags)
        if expected_revision is not None:
            _revision(expected_revision)
            expected = expected_revision
            _validate_authored_revision(public_meta, expected + 1)
            try:
                record = await self.canonical.commit_state(
                    key=key,
                    value=serialized,
                    expected_revision=expected,
                    kind=kind,
                    metadata=self._state_metadata(
                        public_meta=public_meta,
                        tags=normalized_tags,
                        severity=severity,
                        signal=signal,
                        stage=stage,
                    ),
                )
            except StorageConflictError as exc:
                current = await self.canonical.current_state(key=key, kind=kind)
                raise StateSnapshotConflictError(
                    key=key,
                    expected_revision=expected,
                    actual_revision=0 if current is None else current.revision,
                ) from exc
        else:
            record = await self._commit_state_with_retry(
                key=key,
                value=serialized,
                kind=kind,
                public_meta=public_meta,
                tags=normalized_tags,
                severity=severity,
                signal=signal,
                stage=stage,
            )
        return _state_public_event(
            record,
            logical_scope_id=self.memory_scope_id,
            kind=kind,
        )

    async def get_latest_state_record(
        self,
        key: str,
        *,
        tags: list[str] | None = None,
        level: str | None = None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> dict[str, Any] | None:
        """Read the latest public state value and provider revision.

        The lookup uses one indexed canonical current-state row. Public tag filters
        apply to stored snapshot metadata after exact state identity is resolved.

        Examples:
            Read current Agent state:
                ```python
                record = await memory.get_latest_state_record("agent:writer")
                ```

            Require a persisted public tag:
                ```python
                record = await memory.get_latest_state_record(
                    "agent:writer",
                    tags=["verified"],
                    use_persistence=True,
                )
                ```

        Args:
            key: Exact caller-owned state key.
            tags: Optional public tags every matching snapshot must contain.
            level: Deprecated query-level override; only `None` is accepted.
            use_persistence: Retained compatibility flag; canonical state is always durable.
            kind: Exact state family used for canonical namespace isolation.

        Returns:
            dict[str, Any] | None: Value, revision, metadata, and stable state Event ID.

        Notes:
            `use_persistence=False` does not select a cache or fallback path; canonical
            `StateStore` is the sole state authority in both cases.
        """
        _memory_level(level)
        record = await self.canonical.current_state(key=key, kind=kind)
        if record is None or not _state_has_tags(record, _tags(tags)):
            return None
        return _state_record_mapping(record, kind=kind)

    async def get_latest_state(
        self,
        key: str,
        *,
        tags: list[str] | None = None,
        level: str | None = None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> Any | None:
        """Read the latest public value for one exact state key.

        This convenience method delegates to the revision-bearing current-state read
        and returns only the detached JSON value.

        Examples:
            Read a current value:
                ```python
                value = await memory.get_latest_state("agent:writer")
                ```

            Read a tagged custom family:
                ```python
                value = await memory.get_latest_state(
                    "checkpoint",
                    tags=["approved"],
                    kind="workflow.checkpoint",
                )
                ```

        Args:
            key: Exact caller-owned state key.
            tags: Optional public tags every matching snapshot must contain.
            level: Deprecated query-level override; only `None` is accepted.
            use_persistence: Retained compatibility flag; canonical state is always durable.
            kind: Exact state family used for canonical namespace isolation.

        Returns:
            Any | None: Detached current state value or `None` when absent.

        Notes:
            No event scan, hot-cache read, or legacy state convention is used.
        """
        record = await self.get_latest_state_record(
            key,
            tags=tags,
            level=level,
            use_persistence=use_persistence,
            kind=kind,
        )
        return None if record is None else record["value"]

    async def list_state_history(
        self,
        key: str,
        *,
        tags: list[str] | None = None,
        limit: int = 50,
        level: str | None = None,
        kind: str = "state.snapshot",
        use_persistence: bool = False,
    ) -> list[Event]:
        """Read bounded newest-first public state history.

        The provider applies exact scope, namespace, and key before opaque cursor
        pagination. Public tag filtering is applied to the bounded retained page.

        Examples:
            Read recent state revisions:
                ```python
                history = await memory.list_state_history("agent:writer", limit=20)
                ```

            Read a tagged custom state family:
                ```python
                history = await memory.list_state_history(
                    "checkpoint",
                    tags=["approved"],
                    kind="workflow.checkpoint",
                    use_persistence=True,
                )
                ```

        Args:
            key: Exact caller-owned state key.
            tags: Optional public tags every matching snapshot must contain.
            limit: Positive history bound up to 1000 revisions.
            level: Deprecated query-level override; only `None` is accepted.
            kind: Exact state family used for canonical namespace isolation.
            use_persistence: Retained compatibility flag; canonical state is always durable.

        Returns:
            list[Event]: Matching newest-first public state Event DTOs.

        Notes:
            State history is never reconstructed from `EventStore` and never triggers
            a legacy or cache fallback.
        """
        _memory_level(level)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        required_tags = _tags(tags)
        page = await self.canonical.state_history(key=key, kind=kind, limit=limit)
        return [
            _state_public_event(record, logical_scope_id=self.memory_scope_id, kind=kind)
            for record in page.items
            if _state_has_tags(record, required_tags)
        ]

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
            level: Recognized compatibility scope level for the already-bound facade.
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
        _memory_level(level)
        normalized_kinds = _terms("Memory kinds", kinds)
        normalized_tags = _tags(tags)
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
                kinds=normalized_kinds,
                tags=normalized_tags,
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
                kinds=normalized_kinds,
                tags=normalized_tags,
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
        level: str | None = None,
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
            level: Recognized compatibility scope level for the already-bound facade.
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
            level=level,
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

    async def _commit_state_with_retry(
        self,
        *,
        key: str,
        value: Any,
        kind: str,
        public_meta: dict[str, Any],
        tags: tuple[str, ...],
        severity: int,
        signal: float | None,
        stage: str | None,
    ) -> StateRecord:
        last_expected = 0
        for _attempt in range(_MAX_STATE_CAS_RETRIES):
            current = await self.canonical.current_state(key=key, kind=kind)
            expected = 0 if current is None else current.revision
            last_expected = expected
            _validate_authored_revision(public_meta, expected + 1)
            try:
                return await self.canonical.commit_state(
                    key=key,
                    value=value,
                    expected_revision=expected,
                    kind=kind,
                    metadata=self._state_metadata(
                        public_meta=public_meta,
                        tags=tags,
                        severity=severity,
                        signal=signal,
                        stage=stage,
                    ),
                )
            except StorageConflictError:
                continue
        current = await self.canonical.current_state(key=key, kind=kind)
        actual = 0 if current is None else current.revision
        raise StateSnapshotConflictError(
            key=key,
            expected_revision=last_expected,
            actual_revision=actual,
        )

    def _state_metadata(
        self,
        *,
        public_meta: dict[str, Any],
        tags: tuple[str, ...],
        severity: int,
        signal: float | None,
        stage: str | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            _STATE_SERVICE_CONTEXT: {
                "tags": list(tags),
                "severity": severity,
                "signal": signal,
                "stage": stage,
            },
            _STATE_PUBLIC_METADATA: public_meta,
        }
        if self._deprecated_app_id is not None:
            metadata[_COMPATIBILITY_METADATA] = {
                _DEPRECATED_APP_ID: {
                    "value": self._deprecated_app_id,
                    "deprecated": True,
                    "scheduled_removal": "future breaking release",
                }
            }
        return metadata

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


def _state_public_event(
    record: StateRecord,
    *,
    logical_scope_id: str,
    kind: str,
) -> Event:
    public_meta, service, compatibility = _state_metadata_parts(record)
    app_id = _deprecated_app_id(compatibility)
    tags = tuple(service.get("tags") or ())
    severity = service.get("severity")
    signal = service.get("signal")
    stage = service.get("stage")
    return Event(
        event_id=_state_event_id(record),
        ts=record.updated_at.isoformat(),
        run_id=record.scope.run_id or "",
        scope_id=logical_scope_id,
        user_id=record.scope.user_id,
        org_id=record.scope.org_id,
        session_id=record.scope.session_id,
        kind=kind,
        stage=stage if isinstance(stage, str) else None,
        text=f"state:{record.key} " + (_text(_plain(record.value)) or "null"),
        tags=list(_tags(["state", f"state:{record.key}", *tags])),
        data={
            "key": record.key,
            "value": _plain(record.value),
            "meta": {**public_meta, "revision": record.revision},
        },
        graph_id=record.scope.graph_id,
        node_id=record.scope.node_id,
        app_id=app_id,
        agent_id=record.scope.agent_id,
        severity=severity if isinstance(severity, int) else 2,
        signal=float(signal) if isinstance(signal, int | float) else 0.0,
        version=record.schema_version,
    )


def _state_record_mapping(record: StateRecord, *, kind: str) -> dict[str, Any]:
    public_meta, _service, _compatibility = _state_metadata_parts(record)
    return {
        "value": _plain(record.value),
        "revision": record.revision,
        "meta": {**public_meta, "revision": record.revision},
        "event_id": _state_event_id(record),
        "kind": kind,
    }


def _state_metadata_parts(
    record: StateRecord,
) -> tuple[dict[str, Any], dict[str, Any], object]:
    metadata = _plain(record.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("Malformed canonical Memory state metadata")
    unknown = set(metadata) - {
        _STATE_PUBLIC_METADATA,
        _STATE_SERVICE_CONTEXT,
        _COMPATIBILITY_METADATA,
    }
    if unknown:
        raise ValueError("Malformed canonical Memory state metadata fields")
    public = metadata.get(_STATE_PUBLIC_METADATA, {})
    service = metadata.get(_STATE_SERVICE_CONTEXT, {})
    if not isinstance(public, dict) or not isinstance(service, dict):
        raise ValueError("Malformed canonical Memory state metadata sections")
    if set(service) != {"tags", "severity", "signal", "stage"}:
        raise ValueError("Malformed canonical Memory state service metadata")
    tags = service.get("tags")
    if not isinstance(tags, list):
        raise ValueError("Malformed canonical Memory state tags")
    _tags(tags)
    severity = service.get("severity")
    if isinstance(severity, bool) or not isinstance(severity, int) or not 0 <= severity <= 100:
        raise ValueError("Malformed canonical Memory state severity")
    signal = service.get("signal")
    if signal is not None and (isinstance(signal, bool) or not isinstance(signal, int | float)):
        raise ValueError("Malformed canonical Memory state signal")
    stage = service.get("stage")
    if stage is not None and (
        not isinstance(stage, str) or not stage.strip() or stage != stage.strip()
    ):
        raise ValueError("Malformed canonical Memory state stage")
    return public, service, metadata.get(_COMPATIBILITY_METADATA, {})


def _state_has_tags(record: StateRecord, required: tuple[str, ...]) -> bool:
    if not required:
        return True
    _public, service, _compatibility = _state_metadata_parts(record)
    return set(required).issubset(service["tags"])


def _state_event_id(record: StateRecord) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "scope": record.scope.as_filter(),
                "namespace": record.namespace,
                "key": record.key,
                "revision": record.revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"memory-state-{digest}"


def _validate_authored_revision(metadata: dict[str, Any], revision: int) -> None:
    authored = metadata.get("revision")
    if authored is not None and (
        isinstance(authored, bool) or not isinstance(authored, int) or authored != revision
    ):
        raise ValueError("snapshot metadata revision must equal the committed revision")


def _revision(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected_revision must be a non-negative integer")


def _memory_level(value: str | None) -> None:
    if value not in {None, "scope", "session", "run", "user", "org"}:
        raise ValueError("level must be a recognized Memory scope level when supplied")


def _serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _serializable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _serializable(model_dump())
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__repr__": repr(value)}


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


def _tags(values: Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("Memory tags must be a sequence of exact strings, not a string")
    tags = tuple(dict.fromkeys(values or ()))
    if any(not isinstance(tag, str) or not tag.strip() or tag != tag.strip() for tag in tags):
        raise ValueError("Memory tags must contain exact non-empty strings")
    return tags


def _terms(name: str, values: Sequence[str] | None) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of exact strings, not a string")
    terms = tuple(dict.fromkeys(values or ()))
    if any(not isinstance(term, str) or not term.strip() or term != term.strip() for term in terms):
        raise ValueError(f"{name} must contain exact non-empty strings")
    return terms


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
