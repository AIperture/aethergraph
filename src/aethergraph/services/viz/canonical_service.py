"""Canonical visualization persistence over one provider-owned EventStore."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from aethergraph.contracts.services.viz import VizEvent, VizKind, VizMode
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    EventRecord,
    EventStore,
    FrozenJson,
    Page,
    PageRequest,
    SortDirection,
    StorageBundle,
    StorageCapacityError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageScope,
)

_VIZ_KINDS: tuple[VizKind, ...] = ("scalar", "vector", "matrix", "image")
_VIZ_MODES = frozenset({"append", "replace"})
_VIZ_EVENT_PREFIX = "viz."
_COMPATIBILITY_METADATA = "compatibility"
_MAX_QUERY_EVENTS = 10_000
_QUERY_PAGE_SIZE = 500
_DEFAULT_PAGE_REQUEST = PageRequest()


@dataclass(frozen=True, slots=True)
class CanonicalVizEvent:
    """Immutable provider-backed Viz event with an opaque ordering cursor."""

    event_id: str
    cursor: str
    occurred_at: datetime
    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str
    track_id: str
    figure_id: str | None
    viz_kind: VizKind
    step: int
    mode: VizMode
    value: float | None
    vector: tuple[float, ...] | None
    matrix: tuple[tuple[float, ...], ...] | None
    artifact_id: str | None
    meta: Mapping[str, FrozenJson]
    tags: tuple[str, ...]
    deprecated_app_id: str | None = None
    deprecated_client_id: str | None = None


class CanonicalVizService:
    """Persist and read bounded run-scoped Viz events through one EventStore."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
        event_id_factory: Callable[[], str] = lambda: f"viz-{uuid4().hex}",
    ) -> None:
        """Compose canonical Viz behavior from already-open provider dependencies.

        Intro:
            Captures one authoritative EventStore and trusted owner scope. Construction
            performs no I/O, provider selection, open, readiness check, or fallback.

        Examples:
            Compose production Viz persistence:
                ```python
                service = CanonicalVizService(
                    event_store=bundle.events,
                    owner_scope=open_request.owner_scope,
                    clock=open_request.clock.now,
                )
                ```

            Supply deterministic event identities:
                ```python
                service = CanonicalVizService(
                    event_store=fake_events,
                    owner_scope=StorageScope(project_id="project-1"),
                    clock=clock.now,
                    event_id_factory=lambda: "viz-1",
                )
                ```

        Args:
            event_store: Provider-owned canonical runtime EventStore.
            owner_scope: Exact trusted provider ownership scope without run dimensions.
            clock: Timezone-aware UTC timestamp source for events without a timestamp.
            event_id_factory: Source of exact non-empty event identities.

        Returns:
            None: The inactive service is ready without accessing provider resources.

        Notes:
            The owning `StorageComposition` retains lifecycle responsibility. Deprecated
            App/client compatibility metadata never enters canonical scope or indexes.
        """
        validate_storage_owner_scope(owner_scope)
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(event_id_factory):
            raise TypeError("event_id_factory must be callable")
        self._events = event_store
        self.owner_scope = owner_scope
        self._clock = clock
        self._event_id_factory = event_id_factory

    async def append(self, evt: VizEvent) -> None:
        """Commit one validated Viz event in exact owner-plus-run scope.

        Intro:
            Converts the mutable public input into an immutable canonical EventDraft.
            Visualization kind is promoted to `kind="viz.<type>"` for provider-side
            filtering; graph, node, Tool, track, and value data remain payload fields.

        Examples:
            Append a scalar:
                ```python
                await service.append(scalar_event)
                ```

            Append an Artifact-backed image:
                ```python
                await service.append(image_event)
                ```

        Args:
            evt: Public Viz event whose run and value shape must be exact.

        Returns:
            None: The provider has committed the immutable event.

        Notes:
            Optional `app_id` and `client_id` survive only in an explicitly marked
            deprecated compatibility envelope inside payload. They never authorize,
            partition, filter, or index the event.
        """
        if not isinstance(evt, VizEvent):
            raise TypeError("evt must be a VizEvent")
        _validate_owner_provenance(self.owner_scope, evt)
        normalized = _normalize_event(evt)
        occurred_at = _event_time(evt.created_at, self._clock)
        event_id = _exact("event_id", self._event_id_factory())
        scope = merge_storage_scope(self.owner_scope, run_id=normalized.run_id)
        payload: dict[str, Any] = {
            "graph_id": normalized.graph_id,
            "node_id": normalized.node_id,
            "tool_name": normalized.tool_name,
            "tool_version": normalized.tool_version,
            "track_id": normalized.track_id,
            "figure_id": normalized.figure_id,
            "viz_kind": normalized.viz_kind,
            "step": normalized.step,
            "mode": normalized.mode,
            "value": normalized.value,
            "vector": normalized.vector,
            "matrix": normalized.matrix,
            "artifact_id": normalized.artifact_id,
            "meta": normalized.meta,
        }
        compatibility = _compatibility(evt)
        if compatibility:
            payload[_COMPATIBILITY_METADATA] = compatibility
        await self._events.append(
            EventDraft(
                event_id=event_id,
                occurred_at=occurred_at,
                scope=scope,
                kind=_event_kind(normalized.viz_kind),
                topic=normalized.track_id,
                tags=normalized.tags,
                payload=payload,
            )
        )

    async def query_run_page(
        self,
        run_id: str,
        *,
        page: PageRequest = _DEFAULT_PAGE_REQUEST,
        kinds: Sequence[VizKind] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Page[CanonicalVizEvent]:
        """Read one bounded opaque-cursor page of canonical Viz events.

        Intro:
            Applies exact owner-plus-run scope, promoted visualization kinds, and UTC
            time bounds in the provider query before projecting immutable typed rows.

        Examples:
            Read the first run page:
                ```python
                page = await service.query_run_page("run-1")
                ```

            Continue scalar/image history:
                ```python
                page = await service.query_run_page(
                    "run-1",
                    kinds=("scalar", "image"),
                    page=PageRequest(limit=100, cursor=cursor),
                )
                ```

        Args:
            run_id: Exact run identity merged with the trusted owner scope.
            page: Bounded provider-owned opaque cursor request.
            kinds: Optional unique visualization kinds applied before pagination.
            since: Optional inclusive timezone-aware UTC lower timestamp.
            until: Optional inclusive timezone-aware UTC upper timestamp.

        Returns:
            Page[CanonicalVizEvent]: Immutable typed events and next opaque cursor.

        Notes:
            Cursor values are forwarded without parsing. Deprecated compatibility
            metadata is projected but never used as a query dimension.
        """
        if not isinstance(page, PageRequest):
            raise TypeError("page must be a PageRequest")
        scope = merge_storage_scope(self.owner_scope, run_id=_exact("run_id", run_id))
        event_kinds = tuple(_event_kind(kind) for kind in _normalize_kinds(kinds))
        result = await self._events.query(
            EventQuery(
                scope=scope,
                kinds=event_kinds,
                occurred_at_min=_optional_utc("since", since),
                occurred_at_max=_optional_utc("until", until),
                order=SortDirection.ASCENDING,
                page=page,
            )
        )
        return Page(
            items=tuple(_project(record, expected_scope=scope) for record in result.items),
            next_cursor=result.next_cursor,
        )

    async def query_run(
        self,
        run_id: str,
        *,
        kinds: Sequence[VizKind] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        max_events: int = _MAX_QUERY_EVENTS,
    ) -> tuple[CanonicalVizEvent, ...]:
        """Read a complete run response under an explicit fail-closed ceiling.

        Intro:
            Drains bounded opaque provider pages only until the exact query is complete
            or one record exceeds the caller-selected ceiling. Overflow raises instead
            of returning truncated success or allocating an unbounded response.

        Examples:
            Read all bounded Viz events for a run:
                ```python
                events = await service.query_run("run-1")
                ```

            Read recent scalar events under a smaller ceiling:
                ```python
                events = await service.query_run(
                    "run-1",
                    kinds=("scalar",),
                    since=window_start,
                    max_events=500,
                )
                ```

        Args:
            run_id: Exact run identity merged with the trusted owner scope.
            kinds: Optional unique visualization kinds applied before pagination.
            since: Optional inclusive timezone-aware UTC lower timestamp.
            until: Optional inclusive timezone-aware UTC upper timestamp.
            max_events: Positive whole-response ceiling no greater than 10,000.

        Returns:
            tuple[CanonicalVizEvent, ...]: Complete provider-ordered bounded run history.

        Notes:
            A provider returning an empty page with a continuation cursor or reusing a
            cursor fails as an integrity error; the service never falls back to offset.
        """
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be an integer")
        if not 1 <= max_events <= _MAX_QUERY_EVENTS:
            raise ValueError(f"max_events must be between 1 and {_MAX_QUERY_EVENTS}")
        normalized_kinds = _normalize_kinds(kinds)
        rows: list[CanonicalVizEvent] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            remaining_with_overflow_probe = max_events + 1 - len(rows)
            result = await self.query_run_page(
                run_id,
                kinds=normalized_kinds,
                since=since,
                until=until,
                page=PageRequest(
                    limit=min(_QUERY_PAGE_SIZE, remaining_with_overflow_probe),
                    cursor=cursor,
                ),
            )
            rows.extend(result.items)
            if len(rows) > max_events:
                raise StorageCapacityError(
                    f"Viz run query exceeds the explicit {max_events}-event ceiling"
                )
            if result.next_cursor is None:
                return tuple(rows)
            if not result.items:
                raise StorageIntegrityError("Viz provider returned an empty continuation page")
            if result.next_cursor in seen_cursors:
                raise StorageIntegrityError("Viz provider repeated an opaque continuation cursor")
            seen_cursors.add(result.next_cursor)
            cursor = result.next_cursor


def build_canonical_viz_service(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
    event_id_factory: Callable[[], str] = lambda: f"viz-{uuid4().hex}",
) -> CanonicalVizService:
    """Bind canonical Viz behavior to one already-open coherent storage bundle.

    Intro:
        Selects only the typed `bundle.events` field and delegates validation to the
        canonical service constructor. It performs no provider lifecycle operation.

    Examples:
        Bind runtime Viz persistence:
            ```python
            viz = build_canonical_viz_service(
                bundle=bundle,
                owner_scope=open_request.owner_scope,
                clock=open_request.clock.now,
            )
            ```

        Bind deterministic test identities:
            ```python
            viz = build_canonical_viz_service(
                bundle=fake_bundle,
                owner_scope=owner,
                clock=clock.now,
                event_id_factory=lambda: "viz-1",
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Exact trusted provider ownership scope.
        clock: Timezone-aware UTC timestamp source.
        event_id_factory: Source of exact non-empty event identities.

    Returns:
        CanonicalVizService: Inactive service over the bundle's runtime EventStore.

    Notes:
        The returned service does not own bundle close and cannot select another store.
    """
    return CanonicalVizService(
        event_store=bundle.events,
        owner_scope=owner_scope,
        clock=clock,
        event_id_factory=event_id_factory,
    )


@dataclass(frozen=True, slots=True)
class _NormalizedVizEvent:
    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str
    track_id: str
    figure_id: str | None
    viz_kind: VizKind
    step: int
    mode: VizMode
    value: float | None
    vector: tuple[float, ...] | None
    matrix: tuple[tuple[float, ...], ...] | None
    artifact_id: str | None
    meta: Mapping[str, Any]
    tags: tuple[str, ...]


def _normalize_event(evt: VizEvent) -> _NormalizedVizEvent:
    kind = _viz_kind(evt.viz_kind)
    if isinstance(evt.step, bool) or not isinstance(evt.step, int):
        raise TypeError("step must be an integer")
    if evt.mode not in _VIZ_MODES:
        raise StorageConfigurationError("mode must be exactly 'append' or 'replace'")
    value, vector, matrix, artifact_id = _viz_value_shape(
        kind=kind,
        value=evt.value,
        vector=evt.vector,
        matrix=evt.matrix,
        artifact_id=evt.artifact_id,
    )
    if evt.meta is not None and not isinstance(evt.meta, Mapping):
        raise TypeError("meta must be a mapping when supplied")
    tags = tuple(evt.tags or ())
    return _NormalizedVizEvent(
        run_id=_exact("run_id", evt.run_id),
        graph_id=_exact("graph_id", evt.graph_id),
        node_id=_exact("node_id", evt.node_id),
        tool_name=_exact("tool_name", evt.tool_name),
        tool_version=_exact("tool_version", evt.tool_version),
        track_id=_exact("track_id", evt.track_id),
        figure_id=_optional_exact("figure_id", evt.figure_id),
        viz_kind=kind,
        step=evt.step,
        mode=evt.mode,
        value=value,
        vector=vector,
        matrix=matrix,
        artifact_id=artifact_id,
        meta=MappingProxyType(dict(evt.meta or {})),
        tags=tags,
    )


def _validate_owner_provenance(owner_scope: StorageScope, evt: VizEvent) -> None:
    for name in ("org_id", "user_id", "session_id", "run_id", "graph_id", "node_id"):
        owner_value = getattr(owner_scope, name)
        event_value = getattr(evt, name)
        if owner_value is not None and event_value != owner_value:
            raise StorageConfigurationError(f"Viz event conflicts with owner_scope {name}")


def _project(record: EventRecord, *, expected_scope: StorageScope) -> CanonicalVizEvent:
    if record.scope != expected_scope:
        raise StorageIntegrityError("Viz provider returned an event from a different scope")
    if not record.kind.startswith(_VIZ_EVENT_PREFIX):
        raise StorageIntegrityError("Viz provider returned a non-Viz event kind")
    payload = record.payload
    kind = _viz_kind(payload.get("viz_kind"), error=StorageIntegrityError)
    if record.kind != _event_kind(kind):
        raise StorageIntegrityError("Viz event kind conflicts with its payload")
    try:
        normalized = _normalize_event(
            VizEvent(
                run_id=cast(str, record.scope.run_id),
                graph_id=cast(str, payload.get("graph_id")),
                node_id=cast(str, payload.get("node_id")),
                tool_name=cast(str, payload.get("tool_name")),
                tool_version=cast(str, payload.get("tool_version")),
                track_id=cast(str, payload.get("track_id")),
                figure_id=cast(str | None, payload.get("figure_id")),
                viz_kind=kind,
                step=cast(int, payload.get("step")),
                mode=cast(Any, payload.get("mode")),
                value=cast(float | None, payload.get("value")),
                vector=cast(list[float] | None, payload.get("vector")),
                matrix=cast(list[list[float]] | None, payload.get("matrix")),
                artifact_id=cast(str | None, payload.get("artifact_id")),
                meta=cast(dict[str, Any] | None, payload.get("meta")),
                tags=list(record.tags),
            )
        )
    except (TypeError, ValueError) as exc:
        raise StorageIntegrityError("Persisted Viz event payload is malformed") from exc
    app_id, client_id = _project_compatibility(payload.get(_COMPATIBILITY_METADATA))
    return CanonicalVizEvent(
        event_id=record.event_id,
        cursor=record.cursor,
        occurred_at=record.occurred_at,
        run_id=normalized.run_id,
        graph_id=normalized.graph_id,
        node_id=normalized.node_id,
        tool_name=normalized.tool_name,
        tool_version=normalized.tool_version,
        track_id=normalized.track_id,
        figure_id=normalized.figure_id,
        viz_kind=normalized.viz_kind,
        step=normalized.step,
        mode=normalized.mode,
        value=normalized.value,
        vector=normalized.vector,
        matrix=normalized.matrix,
        artifact_id=normalized.artifact_id,
        meta=cast(Mapping[str, FrozenJson], normalized.meta),
        tags=normalized.tags,
        deprecated_app_id=app_id,
        deprecated_client_id=client_id,
    )


def _viz_value_shape(
    *,
    kind: VizKind,
    value: object,
    vector: object,
    matrix: object,
    artifact_id: object,
) -> tuple[
    float | None,
    tuple[float, ...] | None,
    tuple[tuple[float, ...], ...] | None,
    str | None,
]:
    supplied = {
        "value": value is not None,
        "vector": vector is not None,
        "matrix": matrix is not None,
        "artifact_id": artifact_id is not None,
    }
    required = {"scalar": "value", "vector": "vector", "matrix": "matrix", "image": "artifact_id"}[
        kind
    ]
    if not supplied[required] or any(
        present for name, present in supplied.items() if name != required
    ):
        raise StorageConfigurationError(f"{kind} Viz events require only {required}")
    if kind == "scalar":
        return _finite("value", value), None, None, None
    if kind == "vector":
        return None, _number_vector("vector", vector), None, None
    if kind == "matrix":
        return None, None, _number_matrix(matrix), None
    return None, None, None, _exact("artifact_id", artifact_id)


def _number_vector(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray) or not value:
        raise StorageConfigurationError(f"{name} must be a non-empty numeric sequence")
    return tuple(_finite(f"{name}[{index}]", item) for index, item in enumerate(value))


def _number_matrix(value: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray) or not value:
        raise StorageConfigurationError("matrix must be a non-empty row sequence")
    rows = tuple(_number_vector(f"matrix[{index}]", row) for index, row in enumerate(value))
    if len({len(row) for row in rows}) != 1:
        raise StorageConfigurationError("matrix rows must have equal length")
    return rows


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StorageConfigurationError(f"{name} must be finite")
    return result


def _compatibility(evt: VizEvent) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in (("app_id", evt.app_id), ("client_id", evt.client_id)):
        if value is not None:
            result[name] = {
                "value": _exact(name, value),
                "deprecated": True,
                "scheduled_removal": "future breaking release",
            }
    return result


def _project_compatibility(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise StorageIntegrityError("Viz compatibility metadata is malformed")
    unknown = set(value) - {"app_id", "client_id"}
    if unknown:
        raise StorageIntegrityError("Viz compatibility metadata contains unknown keys")
    projected: dict[str, str | None] = {"app_id": None, "client_id": None}
    for name, item in value.items():
        if (
            not isinstance(item, Mapping)
            or item.get("deprecated") is not True
            or item.get("scheduled_removal") != "future breaking release"
        ):
            raise StorageIntegrityError("Viz compatibility metadata is malformed")
        try:
            projected[name] = _exact(name, item.get("value"))
        except (TypeError, StorageConfigurationError) as exc:
            raise StorageIntegrityError("Viz compatibility metadata is malformed") from exc
    return projected["app_id"], projected["client_id"]


def _event_time(value: str | None, clock: Callable[[], datetime]) -> datetime:
    if value is None:
        return _required_utc("clock", clock())
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StorageConfigurationError("created_at must be an exact ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StorageConfigurationError("created_at must be an ISO timestamp") from exc
    return _required_utc("created_at", parsed)


def _required_utc(name: str, value: object) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise StorageConfigurationError(f"{name} must be a timezone-aware UTC datetime")
    return value


def _optional_utc(name: str, value: datetime | None) -> datetime | None:
    return None if value is None else _required_utc(name, value)


def _normalize_kinds(kinds: Sequence[VizKind] | None) -> tuple[VizKind, ...]:
    if kinds is None:
        return _VIZ_KINDS
    if not isinstance(kinds, Sequence) or isinstance(kinds, str | bytes | bytearray):
        raise TypeError("kinds must be a sequence")
    normalized = tuple(_viz_kind(kind) for kind in kinds)
    if not normalized:
        raise StorageConfigurationError("kinds must not be empty when supplied")
    if len(set(normalized)) != len(normalized):
        raise StorageConfigurationError("kinds must not contain duplicates")
    return normalized


def _viz_kind(value: object, *, error: type[Exception] = StorageConfigurationError) -> VizKind:
    if value not in _VIZ_KINDS:
        raise error(f"viz_kind must be one of {_VIZ_KINDS}")
    return cast(VizKind, value)


def _event_kind(kind: VizKind) -> str:
    return f"{_VIZ_EVENT_PREFIX}{kind}"


def _exact(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StorageConfigurationError(f"{name} must be an exact non-empty string")
    return value


def _optional_exact(name: str, value: object) -> str | None:
    return None if value is None else _exact(name, value)
