"""Canonical metering persistence over provider storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    ObservationDraft,
    ObservationQuery,
    ObservationRecord,
    ObservationRepository,
    ObservationSeverity,
    ObservationStatus,
    PageRequest,
    StorageBundle,
    StorageCapacityError,
    StorageIntegrityError,
    StorageScope,
)

_CATEGORY = "metering"
_PRODUCER = "aethergraph.metering"
_MAX_QUERY = 10_000
_PAGE_SIZE = 1_000
_COMPATIBILITY = "compatibility_metadata"
_SERVICE_CONTEXT = "service_context"
_APP_ID = "app_id"
_CLIENT_ID = "client_id"
_SCOPE_FIELDS = ("user_id", "org_id", "run_id", "graph_id", "session_id")
_RESERVED = frozenset(
    {
        "kind",
        "ts",
        "event_id",
        "tags",
        *_SCOPE_FIELDS,
        _APP_ID,
        _CLIENT_ID,
        _COMPATIBILITY,
        _SERVICE_CONTEXT,
    }
)


class CanonicalMeteringStore:
    """Project metering events onto canonical observations."""

    def __init__(
        self,
        *,
        repository: ObservationRepository,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind metering to one canonical observation repository.

        Intro:
            Captures an already-open provider repository and trusted owner scope
            without selecting storage or creating a second event authority.

        Examples:
            Bind the runtime observation repository:
            ```python
            store = CanonicalMeteringStore(
                repository=bundle.observations, owner_scope=scope, clock=clock
            )
            ```

            Bind a deterministic fake:
            ```python
            store = CanonicalMeteringStore(
                repository=fake, owner_scope=test_scope, clock=lambda: fixed_now
            )
            ```

        Args:
            repository: Exact canonical observation repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC default event clock.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            The bundle owns lifecycle; no fallback, opener, or close path exists.
        """
        validate_storage_owner_scope(owner_scope)
        _utc(clock())
        self._repository = repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def append(self, event: dict[str, Any]) -> None:
        """Append one normalized metering observation.

        Intro:
            Promotes query dimensions into canonical fields and keeps deprecated App
            identity only in a marked optional compatibility envelope.

        Examples:
            Record model usage:
            ```python
            await store.append({"kind": "meter.llm", "prompt_tokens": 10})
            ```

            Record an exact retry identity:
            ```python
            await store.append({"event_id": "meter-1", "kind": "meter.run"})
            ```

        Args:
            event: Frozen metering-event-shaped mapping.

        Returns:
            None: The canonical observation was committed.

        Notes:
            Exact caller event IDs inherit provider idempotency; generated IDs are
            unique per append and never derived from deprecated identity.
        """
        if not isinstance(event, dict):
            raise TypeError("metering event must be a dictionary")
        kind = _nonempty("kind", event.get("kind"))
        if not kind.startswith("meter."):
            raise ValueError(f"Metering event kind must start with 'meter.': {kind!r}")
        occurred_at = _event_time(event.get("ts"), self._clock)
        dimensions = {
            name: value for name in _SCOPE_FIELDS if (value := event.get(name)) is not None
        }
        for name, value in dimensions.items():
            _nonempty(name, value)
        scope = merge_storage_scope(self._owner_scope, **dimensions)
        attributes = {key: value for key, value in event.items() if key not in _RESERVED}
        client_id = event.get(_CLIENT_ID)
        if client_id is not None:
            attributes[_SERVICE_CONTEXT] = {_CLIENT_ID: _nonempty(_CLIENT_ID, client_id)}
        app_id = event.get(_APP_ID)
        if app_id is not None:
            attributes[_COMPATIBILITY] = {
                _APP_ID: {
                    "value": _nonempty(_APP_ID, app_id),
                    "deprecated": True,
                    "compatibility_only": True,
                    "scheduled_removal": "future breaking release",
                }
            }
        await self._repository.append_many(
            (
                ObservationDraft(
                    observation_id=str(event.get("event_id") or f"metering-{uuid4().hex}"),
                    category=_CATEGORY,
                    name=kind,
                    summary=kind,
                    occurred_at=occurred_at,
                    scope=scope,
                    status=ObservationStatus.OK,
                    severity=ObservationSeverity.INFO,
                    producer=_PRODUCER,
                    attributes=attributes,
                    retention_class="metering",
                ),
            )
        )

    async def query(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query a bounded metering projection through provider cursors.

        Intro:
            Applies canonical scope, kind, and time filters before pagination and
            fails when the explicit administrative ceiling would be exceeded.

        Examples:
            Query one day's model usage:
            ```python
            rows = await store.query(since=cutoff, kinds=["meter.llm"], limit=1000)
            ```

            Restrict a tenant query:
            ```python
            rows = await store.query(user_id="u1", org_id="o1", limit=100)
            ```

        Args:
            since: Optional inclusive UTC lower time bound.
            until: Optional inclusive UTC upper time bound.
            kinds: Optional exact metering kinds.
            limit: Required maximum result count between one and 10,000.
            user_id: Optional canonical user scope filter.
            org_id: Optional canonical organization scope filter.

        Returns:
            list[dict[str, Any]]: Detached frozen-compatible metering events.

        Notes:
            Unbounded limits, offset cursors, page refetch after overflow, and
            client/App authorization filters are intentionally unsupported.
        """
        if limit is None or isinstance(limit, bool) or not 1 <= limit <= _MAX_QUERY:
            raise ValueError(f"limit must be between 1 and {_MAX_QUERY}")
        dimensions = {
            name: value
            for name, value in (("user_id", user_id), ("org_id", org_id))
            if value is not None
        }
        scope = merge_storage_scope(self._owner_scope, **dimensions)
        names = tuple(dict.fromkeys(kinds or ()))
        records: list[ObservationRecord] = []
        cursor: str | None = None
        while len(records) < limit:
            page = await self._repository.query(
                ObservationQuery(
                    scope=scope,
                    categories=(_CATEGORY,),
                    names=names,
                    occurred_at_or_after=since,
                    occurred_at_or_before=until,
                    page=PageRequest(limit=min(_PAGE_SIZE, limit - len(records)), cursor=cursor),
                )
            )
            records.extend(page.items)
            if page.next_cursor is None:
                return [_project(record) for record in records]
            if not page.items or page.next_cursor == cursor:
                raise StorageIntegrityError("Metering query returned a non-progressing cursor")
            cursor = page.next_cursor
        if cursor is not None:
            raise StorageCapacityError(f"Metering query exceeds explicit limit {limit}")
        return [_project(record) for record in records]


def bind_canonical_metering_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalMeteringStore:
    """Bind metering to the exact observation field of one bundle.

    Intro:
        Constructs the active projection without provider selection, I/O, or
        lifecycle transfer.

    Examples:
        Bind production composition inputs:
        ```python
        store = bind_canonical_metering_store(bundle=bundle, owner_scope=scope, clock=clock)
        ```

        Bind a conformance fake bundle:
        ```python
        store = bind_canonical_metering_store(
            bundle=fake_bundle, owner_scope=test_scope, clock=lambda: fixed_now
        )
        ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC default event clock.

    Returns:
        CanonicalMeteringStore: Exact observation-backed metering projection.

    Notes:
        Binding does not select storage or own bundle close.
    """
    return CanonicalMeteringStore(
        repository=bundle.observations,
        owner_scope=owner_scope,
        clock=clock,
    )


def _project(record: ObservationRecord) -> dict[str, Any]:
    if record.category != _CATEGORY or record.producer != _PRODUCER:
        raise StorageIntegrityError("Canonical metering observation identity mismatch")
    event = dict(record.attributes)
    event["kind"] = record.name
    event["ts"] = record.occurred_at.isoformat()
    for name in _SCOPE_FIELDS:
        if (value := getattr(record.scope, name)) is not None:
            event[name] = value
    service = event.pop(_SERVICE_CONTEXT, None)
    if service is not None:
        if not isinstance(service, Mapping) or set(service) != {_CLIENT_ID}:
            raise StorageIntegrityError("Malformed metering service context")
        event[_CLIENT_ID] = service[_CLIENT_ID]
    compatibility = event.pop(_COMPATIBILITY, None)
    if compatibility is not None:
        if not isinstance(compatibility, Mapping) or set(compatibility) != {_APP_ID}:
            raise StorageIntegrityError("Malformed metering compatibility metadata")
        app = compatibility[_APP_ID]
        if (
            not isinstance(app, Mapping)
            or app.get("deprecated") is not True
            or app.get("compatibility_only") is not True
        ):
            raise StorageIntegrityError("Unmarked metering App compatibility metadata")
        event[_APP_ID] = app.get("value")
    return event


def _event_time(value: object, clock: Callable[[], datetime]) -> datetime:
    if value is None:
        return _utc(clock())
    if isinstance(value, datetime):
        return _utc(value)
    if not isinstance(value, str):
        raise TypeError("metering ts must be an ISO datetime string")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError("metering ts must be an ISO datetime string") from exc


def _nonempty(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("metering timestamps must be timezone-aware UTC")
    return value
