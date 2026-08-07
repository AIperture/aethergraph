from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.storage.event_log import StateSnapshotConflictError
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.services.agent_state import AgentStateConflictError, AgentStateFacade
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.storage.eventlog.fs_event import FSEventLog
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.eventlog.sqlite_event_sync import SQLiteEventLogSync
from aethergraph.storage.memory.event_persist import EventLogPersistence
from aethergraph.storage.memory.fs_persist import FSPersistence
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


def _filesystem_memory(base_dir: Path) -> MemoryFacade:
    return MemoryFacade(
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
        hotlog=KVHotLog(DictKV()),
        persistence=FSPersistence(base_dir=str(base_dir)),
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


def _sqlite_state_cas_process(
    path: str,
    event_id: str,
    barrier,
    results,
) -> None:
    log = SQLiteEventLogSync(path)
    event = _event(event_id, 1000)
    event["scope_id"] = "session-1"
    event["_partition_scope_id"] = "session-1"
    event["data"] = {
        "key": "acct",
        "value": {"writer": event_id},
        "meta": {"revision": 1},
    }
    try:
        barrier.wait(timeout=10)
        log.append_state_snapshot_if_revision(
            event,
            state_key="acct",
            expected_revision=0,
        )
    except StateSnapshotConflictError as exc:
        results.put(("conflict", exc.actual_revision))
    else:
        results.put(("committed", 1))
    finally:
        log.close()


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


@pytest.mark.asyncio
async def test_sqlite_state_cas_rejects_one_independent_writer(tmp_path: Path) -> None:
    path = str(tmp_path / "events.db")
    first_log = SqliteEventLog(path)
    second_log = SqliteEventLog(path)
    try:
        first_memory = _memory(first_log)
        second_memory = _memory(second_log)
        first = _context(first_memory).state("acct", default_factory=dict, level="session")
        second = _context(second_memory).state("acct", default_factory=dict, level="session")
        await asyncio.gather(first.load(), second.load())

        results = await asyncio.gather(
            first.commit({"balance": 1000}, expected_revision=0),
            second.commit({"balance": 950}, expected_revision=0),
            return_exceptions=True,
        )

        assert sum(isinstance(item, AgentStateConflictError) for item in results) == 1
        latest = await first_memory.get_latest_state_record(
            "acct",
            level="session",
            use_persistence=True,
        )
        assert latest is not None
        assert latest["revision"] == 1
        assert latest["value"] in ({"balance": 1000}, {"balance": 950})
        hot_counts = [
            len(
                await memory.query_events(
                    kinds=["state.snapshot"],
                    tags=["state", "state:acct"],
                    use_persistence=False,
                )
            )
            for memory in (first_memory, second_memory)
        ]
        assert sorted(hot_counts) == [0, 1]
    finally:
        first_log._sync._db.close()
        second_log._sync._db.close()


def test_sqlite_state_cas_is_atomic_across_spawned_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    path = str(tmp_path / "events.db")
    processes = [
        context.Process(
            target=_sqlite_state_cas_process,
            args=(path, f"writer-{index}", barrier, results),
        )
        for index in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)

    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = sorted(results.get(timeout=5) for _ in processes)
    assert outcomes == [("committed", 1), ("conflict", 1)]


@pytest.mark.asyncio
async def test_filesystem_eventlog_state_cas_rejects_one_independent_writer(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "events")
    first_memory = _memory(FSEventLog(root))
    second_memory = _memory(FSEventLog(root))
    first = _context(first_memory).state("acct", default_factory=dict, level="session")
    second = _context(second_memory).state("acct", default_factory=dict, level="session")
    await asyncio.gather(first.load(), second.load())

    results = await asyncio.gather(
        first.commit({"balance": 1000}, expected_revision=0),
        second.commit({"balance": 950}, expected_revision=0),
        return_exceptions=True,
    )

    assert sum(isinstance(item, AgentStateConflictError) for item in results) == 1
    latest = await first_memory.get_latest_state_record(
        "acct",
        level="session",
        use_persistence=True,
    )
    assert latest is not None
    assert latest["revision"] == 1


@pytest.mark.asyncio
async def test_filesystem_memory_state_cas_rejects_one_independent_writer(
    tmp_path: Path,
) -> None:
    first_memory = _filesystem_memory(tmp_path)
    second_memory = _filesystem_memory(tmp_path)
    first = _context(first_memory).state("acct", default_factory=dict, level="session")
    second = _context(second_memory).state("acct", default_factory=dict, level="session")
    await asyncio.gather(first.load(), second.load())

    results = await asyncio.gather(
        first.commit({"balance": 1000}, expected_revision=0),
        second.commit({"balance": 950}, expected_revision=0),
        return_exceptions=True,
    )

    assert sum(isinstance(item, AgentStateConflictError) for item in results) == 1
    latest = await first_memory.get_latest_state_record(
        "acct",
        level="session",
        use_persistence=True,
    )
    assert latest is not None
    assert latest["revision"] == 1
