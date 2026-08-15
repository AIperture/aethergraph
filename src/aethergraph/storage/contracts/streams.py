"""Canonical integration-event and runtime-output stream contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .pagination import Page, PageRequest
from .records import FrozenJson, _freeze_mapping, _nonempty, _optional_nonempty, _utc
from .scope import StorageScope


def _strings(values: tuple[str, ...], *, name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _version(value: int) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class InboundEventDraft:
    """Canonical validated Host ingress event before cursor assignment."""

    event_id: str
    deployment_id: str
    route_id: str
    integration_id: str
    external_event_id: str
    received_at: datetime
    scope: StorageScope
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    resource_keys: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_inbound(self)
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))


@dataclass(frozen=True, slots=True, kw_only=True)
class InboundEventRecord:
    """Committed Host ingress event with delivery and pagination cursors."""

    event_id: str
    deployment_id: str
    route_id: str
    integration_id: str
    external_event_id: str
    received_at: datetime
    scope: StorageScope
    delivery_cursor: int
    cursor: str
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    resource_keys: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_inbound(self)
        _delivery_cursor(self.delivery_cursor)
        _nonempty("cursor", self.cursor)
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))


def _validate_inbound(value: InboundEventDraft | InboundEventRecord) -> None:
    for name in (
        "event_id",
        "deployment_id",
        "route_id",
        "integration_id",
        "external_event_id",
    ):
        _nonempty(name, getattr(value, name))
    _utc("received_at", value.received_at)
    value.scope.require("session_id")
    _strings(value.resource_keys, name="resource_keys")
    _version(value.schema_version)


class SemanticEventKind(StrEnum):
    """Exact active semantic-event v2 classification persisted by providers."""

    INPUT_ACCEPTED = "input.accepted"
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    PHASE_CHANGED = "phase.changed"
    PROGRESS_CHANGED = "progress.changed"
    INTERACTION_REQUESTED = "interaction.requested"
    INTERACTION_RESOLVED = "interaction.resolved"
    TOOL_ACTIVITY = "tool.activity"
    ARTIFACT_AVAILABLE = "artifact.available"
    STRUCTURED_OUTPUT = "structured.output"
    WARNING_RAISED = "warning.raised"
    TURN_OUTCOME = "turn.outcome"


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticEventDraft:
    """Canonical semantic integration event before cursor assignment."""

    event_id: str
    deployment_id: str
    turn_id: str
    sequence: int
    producer: str
    occurred_at: datetime
    kind: SemanticEventKind
    scope: StorageScope
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_semantic(self)
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticEventRecord:
    """Committed semantic event with delivery and pagination cursors."""

    event_id: str
    deployment_id: str
    turn_id: str
    sequence: int
    producer: str
    occurred_at: datetime
    kind: SemanticEventKind
    scope: StorageScope
    delivery_cursor: int
    cursor: str
    payload: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_semantic(self)
        _delivery_cursor(self.delivery_cursor)
        _nonempty("cursor", self.cursor)
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, path="payload"))


def _validate_semantic(value: SemanticEventDraft | SemanticEventRecord) -> None:
    for name in ("event_id", "deployment_id", "turn_id", "producer"):
        _nonempty(name, getattr(value, name))
    if isinstance(value.sequence, bool) or value.sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    _utc("occurred_at", value.occurred_at)
    if not isinstance(value.kind, SemanticEventKind):
        raise TypeError("kind must be a SemanticEventKind")
    value.scope.require("session_id")
    _version(value.schema_version)


def _delivery_cursor(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("delivery_cursor must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticEventQuery:
    """Bounded ascending semantic-event session query with opaque cursor."""

    deployment_id: str
    scope: StorageScope
    page: PageRequest = PageRequest()
    kinds: tuple[SemanticEventKind, ...] = ()
    turn_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty("deployment_id", self.deployment_id)
        self.scope.require("session_id")
        if not isinstance(self.kinds, tuple):
            raise TypeError("kinds must be an immutable tuple")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must not contain duplicates")
        if any(not isinstance(value, SemanticEventKind) for value in self.kinds):
            raise TypeError("kinds must contain SemanticEventKind values")
        _optional_nonempty("turn_id", self.turn_id)


class RuntimeOutputStream(StrEnum):
    """Canonical captured process stream."""

    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeOutputFrame:
    """Bounded runtime output frame accepted by a provider-owned sink."""

    output_id: str
    execution_id: str
    scope: StorageScope
    stream: RuntimeOutputStream
    sequence: int
    text: str
    source: str
    tool_name: str | None = None
    partial: bool = False
    truncated: bool = False
    eof: bool = False
    tags: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("output_id", self.output_id)
        _nonempty("execution_id", self.execution_id)
        self.scope.require("run_id", "node_id")
        if not isinstance(self.stream, RuntimeOutputStream):
            raise TypeError("stream must be a RuntimeOutputStream")
        if isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _nonempty("source", self.source)
        _optional_nonempty("tool_name", self.tool_name)
        for name in ("partial", "truncated", "eof"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        _strings(self.tags, name="tags")
        _version(self.schema_version)


class InboundEventRepository(Protocol):
    """Ordered persistence for validated Host ingress events."""

    async def append(self, event: InboundEventDraft) -> InboundEventRecord:
        """Append one validated ingress event before runtime dispatch.

        The provider atomically assigns the positive delivery cursor used by the
        terminal ingress receipt and an opaque record cursor for storage pagination.

        Examples:
            Persist ingress:
                ```python
                stored = await inbound_events.append(event)
                ```

            Retain its cursor:
                ```python
                receipt_cursor = (await inbound_events.append(event)).delivery_cursor
                ```

        Args:
            event: Validated ingress content and materialized resource keys.

        Returns:
            InboundEventRecord: Committed event with delivery and opaque cursors.

        Notes:
            Raw provider payloads and Host schema instances are not accepted. Identity
            conflicts raise `StorageIntegrityError` without alternate persistence.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        event_id: str,
    ) -> InboundEventRecord | None:
        """Read one inbound event by exact scoped identity.

        The lookup does not inspect idempotency rows or retry an external-event alias.

        Examples:
            Read ingress evidence:
                ```python
                event = await inbound_events.get(scope, "ingress-1")
                ```

            Detect absence:
                ```python
                assert await inbound_events.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/session scope constraining access.
            event_id: Exact stable inbound event identity.

        Returns:
            InboundEventRecord | None: Matching event or `None`.

        Notes:
            Deprecated App identity is not a lookup dimension.
        """
        ...


class SemanticEventRepository(Protocol):
    """Ordered semantic events with authored turn-sequence uniqueness."""

    async def append(self, event: SemanticEventDraft) -> SemanticEventRecord:
        """Append one semantic event at its authored turn sequence.

        Identity, `(deployment, session, turn, sequence)`, delivery cursor, and opaque
        record cursor commit together.

        Examples:
            Persist completion:
                ```python
                stored = await semantic_events.append(event)
                ```

            Publish its cursor:
                ```python
                await delivery.publish(event=stored, cursor=stored.delivery_cursor)
                ```

        Args:
            event: Closed provider-neutral semantic event content.

        Returns:
            SemanticEventRecord: Committed event with delivery and opaque cursors.

        Notes:
            Duplicate identity/sequence raises `StorageIntegrityError`; providers do
            not skip, renumber, rewrite, or downgrade semantic events.
        """
        ...

    async def query(self, query: SemanticEventQuery) -> Page[SemanticEventRecord]:
        """Read one bounded ascending deployment/session event page.

        Kind and turn filters apply before cursor pagination for history and reconnect.

        Examples:
            Read history:
                ```python
                page = await semantic_events.query(query)
                ```

            Resume delivery:
                ```python
                page = await semantic_events.query(replace(query, page=next_page))
                ```

        Args:
            query: Exact deployment/session, filters, and opaque page request.

        Returns:
            Page[SemanticEventRecord]: Ascending events and continuation cursor.

        Notes:
            Invalid stored rows fail closed rather than being skipped.
        """
        ...


class RuntimeOutputSink(Protocol):
    """Non-blocking provider-owned sink for bounded runtime output frames."""

    def emit(self, frame: RuntimeOutputFrame) -> None:
        """Accept one frame for ordered durable persistence.

        This synchronous boundary supports stdout/stderr interception and enqueueing.

        Examples:
            Emit stdout:
                ```python
                runtime_output.emit(frame)
                ```

            Emit truncation:
                ```python
                runtime_output.emit(replace(frame, truncated=True))
                ```

        Args:
            frame: Canonical bounded frame with stable execution sequence.

        Returns:
            None: The frame was accepted for persistence.

        Notes:
            Capacity failure is explicit; implementations never redirect to a local
            file, another provider, or an unredacted alternate path.
        """
        ...

    async def flush_execution(self, execution_id: str) -> None:
        """Wait until accepted frames for one execution are durable.

        Other executions may continue enqueueing while this barrier completes.

        Examples:
            Flush a tool:
                ```python
                await runtime_output.flush_execution("execution-1")
                ```

            Flush captured streams:
                ```python
                capture.finish()
                await runtime_output.flush_execution(execution_id)
                ```

        Args:
            execution_id: Exact stable execution identity to flush.

        Returns:
            None: Previously accepted execution frames are durable.

        Notes:
            Persistence failures propagate and are not converted into success.
        """
        ...

    async def flush_run(self, run_id: str) -> None:
        """Wait until accepted frames for one run are durable.

        The barrier covers every run execution accepted before the call.

        Examples:
            Flush before result publication:
                ```python
                await runtime_output.flush_run("run-1")
                ```

            Flush cancellation:
                ```python
                await runtime_output.flush_run(canceled_run_id)
                ```

        Args:
            run_id: Exact stable run identity to flush.

        Returns:
            None: Previously accepted run frames are durable.

        Notes:
            Bundle shutdown owns final sink closure after barriers complete.
        """
        ...
