from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.services.agent_state import AgentStateFacade
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.storage.eventlog.fs_event import FSEventLog
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.memory.event_persist import EventLogPersistence
from aethergraph.storage.memory.hotlog import KVHotLog


class DictKV:
    def __init__(self) -> None:
        self.data = {}

    async def get(self, key, default=None):
        return self.data.get(key, default)

    async def set(self, key, value, ttl_s=None):
        self.data[key] = value


class DictDocs:
    async def put(self, doc_id, obj):
        return None

    async def get(self, doc_id):
        return None

    async def list(self):
        return []


def _memory(log) -> MemoryFacade:
    return MemoryFacade(
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
        hotlog=KVHotLog(DictKV()),
        persistence=EventLogPersistence(log=log, docs=DictDocs()),
        artifact_store=SimpleNamespace(),
    )


def _context(memory: MemoryFacade) -> NodeContext:
    return NodeContext(
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
        services=NodeServices(
            channels=SimpleNamespace(),
            continuation_store=SimpleNamespace(),
            artifact_store=SimpleNamespace(),
            memory_facade=memory,
            agent_state=AgentStateFacade(memory=memory),
        ),
    )


def _event(event_id: str, balance: int) -> dict:
    return {
        "event_id": event_id,
        "ts": 1.0,
        "scope_id": "scope-1",
        "_partition_scope_id": "scope-1",
        "run_id": "run-1",
        "kind": "state.snapshot",
        "tags": ["state", "state:acct"],
        "data": {"value": {"balance": balance}},
    }


@pytest.mark.asyncio
async def test_context_state_load_returns_latest_persisted_snapshot(tmp_path: Path) -> None:
    log = SqliteEventLog(str(tmp_path / "events.db"))
    try:
        memory = _memory(log)
        writer = _context(memory).state("acct", default_factory=dict, level="session")

        await writer.commit({"total_income": 1000, "balance": 1000})
        await writer.commit({"total_income": 1000, "balance": 950})

        loaded = await _context(memory).state("acct", default_factory=dict, level="session").load()

        assert loaded == {"total_income": 1000, "balance": 950}
    finally:
        log._sync._db.close()


@pytest.mark.asyncio
async def test_sqlite_event_query_defaults_to_newest_first(tmp_path: Path) -> None:
    log = SqliteEventLog(str(tmp_path / "events.db"))
    try:
        await log.append(_event("first", 1000))
        await log.append(_event("second", 950))

        newest = await log.query(
            scope_id="scope-1",
            kinds=["state.snapshot"],
            tags=["state", "state:acct"],
            limit=1,
        )
        oldest = await log.query(
            scope_id="scope-1",
            kinds=["state.snapshot"],
            tags=["state", "state:acct"],
            limit=1,
            order_dir="asc",
        )

        assert [row["event_id"] for row in newest] == ["second"]
        assert [row["event_id"] for row in oldest] == ["first"]
    finally:
        log._sync._db.close()


@pytest.mark.asyncio
async def test_fs_event_query_defaults_to_newest_first(tmp_path: Path) -> None:
    log = FSEventLog(str(tmp_path / "events"))

    await log.append(_event("first", 1000))
    await log.append(_event("second", 950))

    newest = await log.query(
        scope_id="scope-1",
        kinds=["state.snapshot"],
        tags=["state", "state:acct"],
        limit=1,
    )
    oldest = await log.query(
        scope_id="scope-1",
        kinds=["state.snapshot"],
        tags=["state", "state:acct"],
        limit=1,
        order_dir="asc",
    )

    assert [row["event_id"] for row in newest] == ["second"]
    assert [row["event_id"] for row in oldest] == ["first"]


@pytest.mark.asyncio
async def test_hotlog_and_persistence_limit_one_agree_on_latest(tmp_path: Path) -> None:
    log = SqliteEventLog(str(tmp_path / "events.db"))
    try:
        memory = _memory(log)

        await memory.append_state_snapshot("acct", {"balance": 1000})
        await memory.append_state_snapshot("acct", {"balance": 950})

        hot = await memory.query_events(
            kinds=["state.snapshot"],
            tags=["state", "state:acct"],
            limit=1,
            use_persistence=False,
        )
        persisted = await memory.query_events(
            kinds=["state.snapshot"],
            tags=["state", "state:acct"],
            limit=1,
            use_persistence=True,
        )

        assert hot[0].event_id == persisted[0].event_id
        expected_data = {"key": "acct", "value": {"balance": 950}, "meta": {}}
        assert hot[0].data == persisted[0].data == expected_data
    finally:
        log._sync._db.close()
