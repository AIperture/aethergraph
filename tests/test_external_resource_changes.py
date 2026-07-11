from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.memory import ExternalResourceChangedEvent
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.memory.event_persist import EventLogPersistence
from aethergraph.storage.memory.hotlog import KVHotLog


class _DictKV:
    def __init__(self) -> None:
        self.data = {}

    async def get(self, key, default=None):
        return self.data.get(key, default)

    async def set(self, key, value, ttl_s=None):
        del ttl_s
        self.data[key] = value


class _DictDocs:
    async def put(self, doc_id, obj):
        del doc_id, obj

    async def get(self, doc_id):
        del doc_id
        return None

    async def list(self):
        return []


def _memory(log: SqliteEventLog) -> MemoryFacade:
    return MemoryFacade(
        run_id="outbox-ingestion",
        session_id="session-1",
        graph_id=None,
        node_id=None,
        hotlog=KVHotLog(_DictKV()),
        persistence=EventLogPersistence(log=log, docs=_DictDocs()),
        artifact_store=SimpleNamespace(),
    )


def _change(*, event_id: str, source_sequence: int) -> ExternalResourceChangedEvent:
    return ExternalResourceChangedEvent(
        event_id=event_id,
        scope_id="session-1",
        session_id="session-1",
        source_sequence=source_sequence,
        resource_key="design_config:project-42",
        resource_kind="design_config",
        previous_revision=str(source_sequence - 1),
        revision=str(source_sequence),
        changed_fields=("lens.aperture",),
        summary="Aperture changed in the UI.",
        source="design_ui",
        recorded_at=f"2026-07-10T20:00:{source_sequence:02d}Z",
    )


def test_external_resource_contract_rejects_content_and_unstable_identity() -> None:
    with pytest.raises(ValueError, match="unknown fields: config"):
        ExternalResourceChangedEvent.from_dict(
            {
                "kind": "external.resource.changed",
                "config": {"large": True},
            }
        )

    with pytest.raises(ValueError, match="namespaced identity"):
        ExternalResourceChangedEvent(
            event_id="evt-1",
            scope_id="session-1",
            session_id="session-1",
            source_sequence=1,
            resource_key="project-42",
            resource_kind="design_config",
            revision="1",
            source="design_ui",
            recorded_at="2026-07-10T20:00:01Z",
        )


@pytest.mark.asyncio
async def test_committed_outbox_events_preserve_identity_and_source_order(
    tmp_path: Path,
) -> None:
    log = SqliteEventLog(str(tmp_path / "events.db"))
    try:
        memory = _memory(log)

        first = await memory.append_external_resource_change(
            _change(event_id="outbox-1", source_sequence=1)
        )
        second = await memory.append_external_resource_change(
            _change(event_id="outbox-2", source_sequence=2)
        )
        persisted = await memory.query_events(
            kinds=["external.resource.changed"],
            tags=["external_resource", "external_source:design_ui"],
            limit=10,
            use_persistence=True,
            order_dir="asc",
        )
        fetched = await memory.get_event("outbox-2")

        assert first.event_id == "outbox-1"
        assert second.event_id == "outbox-2"
        assert [event.event_id for event in persisted] == ["outbox-1", "outbox-2"]
        assert [event.data["source_sequence"] for event in persisted] == [1, 2]
        assert all("config" not in event.data for event in persisted)
        assert fetched is not None
        assert fetched.data["revision"] == "2"
    finally:
        log._sync._db.close()


@pytest.mark.asyncio
async def test_external_resource_ingestion_rejects_wrong_scope_without_persisting(
    tmp_path: Path,
) -> None:
    log = SqliteEventLog(str(tmp_path / "events.db"))
    try:
        memory = _memory(log)
        change = _change(event_id="outbox-1", source_sequence=1)
        wrong_scope = ExternalResourceChangedEvent.from_dict(
            {**change.to_dict(), "scope_id": "session-other"}
        )

        with pytest.raises(ValueError, match="scope_id does not match"):
            await memory.append_external_resource_change(wrong_scope)

        persisted = await memory.query_events(
            kinds=["external.resource.changed"],
            limit=10,
            use_persistence=True,
        )
        assert persisted == []
    finally:
        log._sync._db.close()
