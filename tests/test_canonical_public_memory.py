from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.agent_state import AgentStateFacade
from aethergraph.services.memory import (
    CanonicalMemoryFacadeFactory,
    CanonicalPublicMemoryFacade,
)
from aethergraph.storage.contracts import (
    EventQuery,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://tests/public-memory"
_SECRET = b"canonical-public-memory-secret-32b"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 11, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(seconds=1)
        return value


class _Identities:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"event-{self.value}"


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


@dataclass
class _DemoState:
    count: int = 0

    @classmethod
    def from_dict(cls, value: dict | None) -> _DemoState:
        return cls(count=int((value or {}).get("count") or 0))


def _open_bundle(root: Path, clock: _Clock):
    return LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    ).open(
        StorageOpenRequest(
            workspace_id="public-memory-tests",
            workspace_root=root.resolve(),
            owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=clock,
            secrets=_Secrets(),
        )
    )


@pytest.mark.asyncio
async def test_public_memory_projects_events_without_identity_partition_aliases(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    factory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
        event_id_factory=_Identities(),
    )
    memory = factory.for_public_execution(
        StorageScope(
            session_id="session-1",
            run_id="run-1",
            graph_id="graph-1",
            node_id="node-1",
            agent_id="writer",
        ),
        logical_scope_id="session:session-1",
        deprecated_app_id="app-legacy",
    )
    try:
        first = await memory.append_chat_turn("user", "hello", tags=["chat", "important"])
        second = await memory.append_event(
            kind="tool_result",
            data={"count": 3},
            tool="search",
            inputs=[{"query": "hello"}],
            outputs=[{"count": 3}],
        )

        assert isinstance(memory, CanonicalPublicMemoryFacade)
        assert first.event_id == "event-1"
        assert first.scope_id == "session:session-1"
        assert first.app_id == "app-legacy"
        assert first.tags == ["chat", "important"]
        assert second.topic == second.tool == "search"
        assert second.inputs == [{"query": "hello"}]
        assert second.outputs == [{"count": 3}]
        assert await memory.get_event("event-1") == first

        durable = await memory.query_events(
            use_persistence=True,
            limit=1,
            offset=1,
            order_dir="asc",
            return_event=False,
        )
        assert [row["event_id"] for row in durable] == ["event-2"]
        assert durable[0]["app_id"] == "app-legacy"
        hot = await memory.recent_events(limit=2)
        assert [event.event_id for event in hot] == ["event-2", "event-1"]

        canonical = await bundle.memory_events.query(EventQuery(scope=memory.scope))
        assert canonical.items[0].scope.as_filter() == {
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "session_id": "session-1",
            "run_id": "run-1",
            "graph_id": "graph-1",
            "node_id": "node-1",
            "agent_id": "writer",
        }
        assert "app_id" not in canonical.items[0].scope.as_filter()
        assert canonical.items[0].payload["compatibility_metadata"]["app_id"] == {
            "value": "app-legacy",
            "deprecated": True,
            "scheduled_removal": "future breaking release",
        }
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_query_and_alias_failures_are_direct(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
        event_id_factory=_Identities(),
    ).for_public_execution(
        StorageScope(session_id="session-1", run_id="run-1"),
        logical_scope_id="run:run-1",
    )
    try:
        with pytest.raises(ValueError, match="match"):
            await memory.append_event(
                kind="tool_result",
                data={},
                topic="search",
                tool="other",
            )
        with pytest.raises(ValueError, match="client_id"):
            await memory.query_events(client_id="client-1")
        with pytest.raises(ValueError, match="run_id"):
            await memory.query_events(run_id="run-2")
        with pytest.raises(ValueError, match="10000"):
            await memory.query_events(limit=10, offset=10_000)
        with pytest.raises(ValueError, match="timezone"):
            await memory.query_events(since="2026-08-16T11:00:00")
        with pytest.raises(ValueError, match="order_dir"):
            await memory.query_events(order_dir="sideways")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="tags"):
            await memory.append_event(kind="invalid", data={}, tags=[" bad"])
        assert await memory.get_event("missing") is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_state_uses_state_current_history_without_event_duplication(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
        event_id_factory=_Identities(),
    ).for_public_execution(
        StorageScope(session_id="session-1", run_id="run-1", agent_id="writer"),
        logical_scope_id="session:session-1",
        deprecated_app_id="app-legacy",
    )
    try:
        first = await memory.append_state_snapshot(
            "agent:writer",
            SimpleNamespace(value=1),
            tags=["state", "verified"],
            meta={"revision": 1, "reason": "create"},
        )
        second = await memory.append_state_snapshot(
            "agent:writer",
            {"value": 2},
            tags=["verified"],
            meta={"revision": 2, "reason": "advance"},
            expected_revision=1,
        )

        assert first.data["meta"]["revision"] == 1
        assert second.data == {
            "key": "agent:writer",
            "value": {"value": 2},
            "meta": {"revision": 2, "reason": "advance"},
        }
        assert second.tags == ["state", "state:agent:writer", "verified"]
        assert second.app_id == "app-legacy"
        assert await memory.get_latest_state("agent:writer") == {"value": 2}
        latest = await memory.get_latest_state_record("agent:writer", tags=["verified"])
        assert latest == {
            "value": {"value": 2},
            "revision": 2,
            "meta": {"revision": 2, "reason": "advance"},
            "event_id": second.event_id,
            "kind": "state.snapshot",
        }
        assert await memory.get_latest_state_record("agent:writer", tags=["missing"]) is None
        history = await memory.list_state_history("agent:writer", limit=10)
        assert [event.data["meta"]["revision"] for event in history] == [2, 1]
        assert [event.event_id for event in history] == [second.event_id, first.event_id]

        event_page = await bundle.memory_events.query(EventQuery(scope=memory.scope))
        assert event_page.items == ()
        current = await bundle.state.get(
            memory.scope,
            "memory.state.state.snapshot",
            "agent:writer",
        )
        assert current is not None
        assert "app_id" not in current.scope.as_filter()
        assert current.metadata["compatibility_metadata"]["app_id"]["deprecated"] is True
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_state_conflicts_preserve_public_error(tmp_path: Path) -> None:
    from aethergraph.contracts.storage.event_log import StateSnapshotConflictError

    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-1"),
        logical_scope_id="run:run-1",
    )
    try:
        await memory.append_state_snapshot("checkpoint", {"step": 1}, expected_revision=0)
        with pytest.raises(StateSnapshotConflictError) as conflict:
            await memory.append_state_snapshot(
                "checkpoint",
                {"step": 2},
                expected_revision=0,
            )
        assert conflict.value.expected_revision == 0
        assert conflict.value.actual_revision == 1
        with pytest.raises(ValueError, match="committed revision"):
            await memory.append_state_snapshot(
                "checkpoint",
                {"step": 2},
                expected_revision=1,
                meta={"revision": 9},
            )
        with pytest.raises(ValueError, match="level"):
            await memory.get_latest_state("checkpoint", level="application")
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_unconditional_state_writes_retry_same_store_contention(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-1"),
        logical_scope_id="run:run-1",
    )
    try:
        events = await asyncio.gather(
            *(memory.append_state_snapshot("counter", {"writer": index}) for index in range(8))
        )

        revisions = sorted(event.data["meta"]["revision"] for event in events)
        assert revisions == list(range(1, 9))
        history = await memory.list_state_history("counter", limit=8)
        assert [event.data["meta"]["revision"] for event in history] == list(range(8, 0, -1))
        event_page = await bundle.memory_events.query(EventQuery(scope=memory.scope))
        assert event_page.items == ()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_preserves_active_agent_state_facade_revision_behavior(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        clock=clock.now,
    ).for_public_execution(
        StorageScope(session_id="session-1", run_id="run-1", agent_id="writer"),
        logical_scope_id="session:session-1",
    )
    try:
        first = AgentStateFacade(memory=memory).bind(  # type: ignore[arg-type]
            key="demo",
            model=_DemoState,
            default_factory=_DemoState,
        )
        initial = await first.load()
        initial.count = 1
        await first.commit(initial, expected_revision=0)

        second = AgentStateFacade(memory=memory).bind(  # type: ignore[arg-type]
            key="demo",
            model=_DemoState,
            default_factory=_DemoState,
        )
        loaded = await second.load()
        await second.commit(_DemoState(count=2), expected_revision=second.revision)

        assert loaded == _DemoState(count=1)
        assert second.revision == 2
        assert await memory.get_latest_state("demo") == {"count": 2}
    finally:
        await bundle.close()


def test_public_memory_docstrings_and_surface_are_explicit() -> None:
    for member in (
        CanonicalPublicMemoryFacade.__init__,
        CanonicalPublicMemoryFacade.append_event,
        CanonicalPublicMemoryFacade.append_chat_turn,
        CanonicalPublicMemoryFacade.get_event,
        CanonicalPublicMemoryFacade.append_state_snapshot,
        CanonicalPublicMemoryFacade.get_latest_state_record,
        CanonicalPublicMemoryFacade.get_latest_state,
        CanonicalPublicMemoryFacade.list_state_history,
        CanonicalPublicMemoryFacade.query_events,
        CanonicalPublicMemoryFacade.recent_events,
        CanonicalPublicMemoryFacade.event_to_dict,
        CanonicalMemoryFacadeFactory.for_public_execution,
    ):
        docstring = inspect.getdoc(member) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

    signature = inspect.signature(CanonicalPublicMemoryFacade)
    assert "deprecated_app_id" in signature.parameters
    assert "app_id" not in signature.parameters
    source = inspect.getsource(CanonicalPublicMemoryFacade)
    assert "_row_id" not in source
    assert "after_id" not in source
    assert "before_id" not in source
    assert "EventLog" not in source
