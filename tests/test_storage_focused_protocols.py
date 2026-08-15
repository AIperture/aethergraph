from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
import inspect
from typing import get_type_hints

import pytest
from storage_conformance.suite import (
    check_event_store_conformance,
    check_state_store_conformance,
)

from aethergraph.storage.contracts import (
    ArtifactRepository,
    BlobRange,
    BlobStore,
    EventDraft,
    EventQuery,
    EventRecord,
    EventStore,
    FrozenJson,
    Page,
    PageRequest,
    SearchBackend,
    SortDirection,
    StateHistoryQuery,
    StateRecord,
    StateStore,
    StorageBundle,
    StorageConflictError,
    StorageScope,
)


def _scope_key(scope: StorageScope) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(scope.as_filter().items()))


class _EventStore:
    def __init__(self) -> None:
        self.rows: list[EventRecord] = []

    async def append(self, event: EventDraft) -> EventRecord:
        existing = next((row for row in self.rows if row.event_id == event.event_id), None)
        if existing is not None:
            return existing
        row = EventRecord(
            **{name: getattr(event, name) for name in event.__dataclass_fields__},
            cursor=f"cursor-{len(self.rows) + 1}",
        )
        self.rows.append(row)
        return row

    async def append_many(self, events: tuple[EventDraft, ...]) -> tuple[EventRecord, ...]:
        return tuple([await self.append(event) for event in events])

    async def get(self, scope: StorageScope, event_id: str) -> EventRecord | None:
        return next(
            (
                row
                for row in self.rows
                if row.event_id == event_id and _scope_key(row.scope) == _scope_key(scope)
            ),
            None,
        )

    async def query(self, query: EventQuery) -> Page[EventRecord]:
        rows = [row for row in self.rows if _scope_key(row.scope) == _scope_key(query.scope)]
        if query.kinds:
            rows = [row for row in rows if row.kind in query.kinds]
        rows.sort(key=lambda row: int(row.cursor.split("-")[-1]), reverse=True)
        if query.order is SortDirection.ASCENDING:
            rows.reverse()
        start = int(query.page.cursor.split("-")[-1]) if query.page.cursor else 0
        selected = tuple(rows[start : start + query.page.limit])
        next_index = start + len(selected)
        next_cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=next_cursor)


class _StateStore:
    def __init__(self) -> None:
        self.current: dict[tuple, StateRecord] = {}
        self.revisions: dict[tuple, list[StateRecord]] = {}

    @staticmethod
    def _key(scope: StorageScope, namespace: str, key: str) -> tuple:
        return (_scope_key(scope), namespace, key)

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> StateRecord | None:
        return self.current.get(self._key(scope, namespace, key))

    async def get_many(
        self,
        scope: StorageScope,
        namespace: str,
        keys: tuple[str, ...],
    ) -> tuple[StateRecord | None, ...]:
        return tuple([await self.get(scope, namespace, key) for key in keys])

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        metadata: Mapping[str, FrozenJson],
    ) -> StateRecord:
        storage_key = self._key(scope, namespace, key)
        current = self.current.get(storage_key)
        actual = current.revision if current else 0
        if actual != expected_revision:
            raise StorageConflictError(f"expected revision {expected_revision}, found {actual}")
        record = StateRecord(
            namespace=namespace,
            key=key,
            value=value,
            revision=actual + 1,
            scope=scope,
            updated_at=datetime.now(UTC),
            metadata=metadata,
        )
        self.current[storage_key] = record
        self.revisions.setdefault(storage_key, []).append(record)
        return record

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        storage_key = self._key(scope, namespace, key)
        current = self.current.get(storage_key)
        if current is None:
            if expected_revision == 0:
                return False
            raise StorageConflictError("state is absent")
        if current.revision != expected_revision:
            raise StorageConflictError("state revision changed")
        del self.current[storage_key]
        return True

    async def history(self, query: StateHistoryQuery) -> Page[StateRecord]:
        rows = list(self.revisions.get(self._key(query.scope, query.namespace, query.key), []))
        rows.sort(key=lambda row: row.revision, reverse=True)
        if query.order is SortDirection.ASCENDING:
            rows.reverse()
        start = int(query.page.cursor.split("-")[-1]) if query.page.cursor else 0
        selected = tuple(rows[start : start + query.page.limit])
        next_index = start + len(selected)
        cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=cursor)


@pytest.mark.asyncio
async def test_fake_event_store_passes_shared_conformance_suite() -> None:
    await check_event_store_conformance(_EventStore())


@pytest.mark.asyncio
async def test_fake_state_store_passes_shared_conformance_suite() -> None:
    await check_state_store_conformance(_StateStore())


def test_query_and_blob_range_validation_is_explicit() -> None:
    scope = StorageScope(tenant_id="tenant-1")
    query = EventQuery(
        scope=scope,
        page=PageRequest(limit=25),
        kinds=("memory.event",),
        order=SortDirection.ASCENDING,
    )

    assert replace(query, page=PageRequest(cursor="opaque")).page.cursor == "opaque"
    assert BlobRange(start=0, end=10).end == 10
    with pytest.raises(ValueError, match="greater"):
        BlobRange(start=10, end=10)
    with pytest.raises(ValueError, match="duplicates"):
        EventQuery(scope=scope, tags=("memory", "memory"))


def test_bundle_exposes_only_typed_canonical_high_frequency_store_fields() -> None:
    hints = get_type_hints(StorageBundle)

    assert hints["events"] is EventStore
    assert hints["memory_events"] is EventStore
    assert hints["state"] is StateStore
    assert hints["blobs"] is BlobStore
    assert hints["artifacts"] is ArtifactRepository
    assert hints["search"] is SearchBackend
    assert "ext_services" not in hints
    assert "stores" not in hints


def test_public_protocol_docstrings_follow_required_section_order() -> None:
    protocols = (EventStore, StateStore, BlobStore, ArtifactRepository, SearchBackend)
    required = ("Examples:", "Args:", "Returns:", "Notes:")

    for protocol in protocols:
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)
