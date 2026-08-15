from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.services.agent_state import AgentStateConflictError, AgentStateFacade


@dataclass
class DemoState:
    count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"count": self.count}

    @classmethod
    def from_dict(cls, data: dict | None) -> DemoState:
        return cls(count=int((data or {}).get("count") or 0))


@dataclass
class CanonicalState:
    pairs: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"pairs": dict(self.pairs)}


class FakeMemory:
    def __init__(self) -> None:
        self.memory_scope_id = "session-1"
        self.session_id = "session-1"
        self.latest = None
        self.latest_meta = {}
        self.record_state_calls = []
        self.record_calls = []
        self.history_calls = []
        self.search_calls = []
        self.external_resource_changes = []

    async def get_latest_state_record(self, key, **kwargs):
        if self.latest is None:
            return None
        return {
            "value": self.latest,
            "revision": int(self.latest_meta.get("revision") or 0),
            "meta": dict(self.latest_meta),
            "event_id": f"state-{len(self.record_state_calls)}",
        }

    async def append_state_snapshot(self, **kwargs):
        self.record_state_calls.append(kwargs)
        self.latest = kwargs["value"]
        self.latest_meta = dict(kwargs.get("meta") or {})
        return SimpleNamespace(event_id=f"state-{len(self.record_state_calls)}")

    async def record(self, **kwargs):
        self.record_calls.append(kwargs)
        return SimpleNamespace(event_id=f"event-{len(self.record_calls)}")

    async def append_external_resource_change(self, change):
        self.external_resource_changes.append(change)
        return SimpleNamespace(event_id=change.event_id, data=change.to_dict())

    async def list_state_history(self, key, **kwargs):
        self.history_calls.append((key, kwargs))
        return ["history"]

    async def search_state(self, **kwargs):
        self.search_calls.append(kwargs)
        return ["search"]


@pytest.mark.asyncio
async def test_agent_state_load_returns_default_and_commit_records_once() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(
        key="demo",
        model=DemoState,
        default_factory=DemoState,
        tags=["demo-tag"],
        meta={"agent": "demo-agent"},
    )

    state = await store.load()
    assert state == DemoState()

    state.count = 2
    await store.commit(state, reason="unit_test", stage_id="stage-a")

    assert len(memory.record_state_calls) == 1
    call = memory.record_state_calls[0]
    assert call["key"] == "demo"
    assert call["value"] == {"count": 2}
    assert call["tags"] == ["demo-tag"]
    assert call["meta"]["agent"] == "demo-agent"
    assert call["meta"]["reason"] == "unit_test"
    assert call["meta"]["stage_id"] == "stage-a"
    assert call["stage"] == "stage-a"


@pytest.mark.asyncio
async def test_agent_state_dataclass_round_trip_uses_from_dict() -> None:
    memory = FakeMemory()
    memory.latest = {"count": 7}
    store = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)

    state = await store.load()

    assert state == DemoState(count=7)


@pytest.mark.asyncio
async def test_agent_state_prefers_canonical_to_dict_over_dataclass_internals() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(key="canonical")

    await store.commit(CanonicalState(pairs=(("docs_read", "digest-1"),)))

    assert memory.record_state_calls[0]["value"] == {"pairs": {"docs_read": "digest-1"}}


@pytest.mark.asyncio
async def test_agent_state_local_backend_does_not_write_memory() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(
        key="demo",
        model=DemoState,
        default_factory=DemoState,
        backend="local",
    )

    state = await store.load()
    state.count = 3
    await store.commit(state, reason="local")
    await store.emit_change(
        event_id="local-1",
        source_sequence=1,
        revision="1",
        recorded_at="2026-07-10T20:00:01Z",
        reason="local-change",
    )

    assert memory.record_state_calls == []
    assert memory.record_calls == []
    assert memory.external_resource_changes == []


@pytest.mark.asyncio
async def test_agent_state_hybrid_does_not_write_until_commit() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(
        key="demo",
        model=DemoState,
        default_factory=DemoState,
        backend="hybrid",
    )

    state = await store.load()
    state.count = 5

    assert memory.record_state_calls == []
    await store.commit(state)
    assert len(memory.record_state_calls) == 1


@pytest.mark.asyncio
async def test_agent_state_force_load_refreshes_cached_hybrid_state() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(
        key="demo",
        model=DemoState,
        default_factory=DemoState,
        backend="hybrid",
    )

    memory.latest = {"count": 1}
    first = await store.load()
    assert first == DemoState(count=1)

    memory.latest = {"count": 2}
    cached = await store.load()
    assert cached == DemoState(count=1)

    refreshed = await store.load(force=True)
    assert refreshed == DemoState(count=2)


@pytest.mark.asyncio
async def test_agent_state_commit_rejects_stale_expected_revision_before_write() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)
    state = await store.load()

    state.count = 1
    await store.commit(state, expected_revision=0)

    with pytest.raises(AgentStateConflictError) as conflict:
        await store.commit(DemoState(count=2), expected_revision=0)

    assert conflict.value.expected_revision == 0
    assert conflict.value.actual_revision == 1
    assert store.revision == 1
    assert len(memory.record_state_calls) == 1


@pytest.mark.asyncio
async def test_agent_state_concurrent_cas_allows_only_one_shared_handle_writer() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)
    await store.load()

    results = await asyncio.gather(
        store.commit(DemoState(count=1), expected_revision=0),
        store.commit(DemoState(count=2), expected_revision=0),
        return_exceptions=True,
    )

    assert sum(isinstance(item, AgentStateConflictError) for item in results) == 1
    assert len(memory.record_state_calls) == 1
    assert store.revision == 1


@pytest.mark.asyncio
async def test_agent_state_new_handle_hydrates_persisted_snapshot_revision() -> None:
    memory = FakeMemory()
    first = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)
    await first.commit(DemoState(count=1), expected_revision=0)

    second = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)
    loaded = await second.load()
    await second.commit(DemoState(count=2), expected_revision=second.revision)

    assert loaded == DemoState(count=1)
    assert second.revision == 2
    assert memory.latest_meta["revision"] == 2


@pytest.mark.asyncio
async def test_agent_state_bind_distinguishes_scope_configuration() -> None:
    memory = FakeMemory()
    facade = AgentStateFacade(memory=memory)

    session_store = facade.bind(key="demo", model=DemoState, level="session")
    run_store = facade.bind(key="demo", model=DemoState, level="run")

    assert session_store is not run_store
    assert session_store.level == "session"
    assert run_store.level == "run"


@pytest.mark.asyncio
async def test_agent_state_emit_change_uses_external_resource_contract() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)

    await store.emit_change(
        event_id="state-outbox-1",
        source_sequence=1,
        revision="1",
        recorded_at="2026-07-10T20:00:01Z",
        reason="stage_started",
        patch={"pipeline.active_stage_id": "stage-a"},
        summary="Stage started.",
    )

    assert memory.record_calls == []
    assert len(memory.external_resource_changes) == 1
    change = memory.external_resource_changes[0]
    assert change.to_dict() == {
        "kind": "external.resource.changed",
        "event_id": "state-outbox-1",
        "scope_id": "session-1",
        "session_id": "session-1",
        "source_sequence": 1,
        "resource_key": "agent_state:demo",
        "resource_kind": "agent_state",
        "revision": "1",
        "source": "agent_state",
        "recorded_at": "2026-07-10T20:00:01Z",
        "changed_fields": ["pipeline.active_stage_id"],
        "summary": "Stage started.",
    }
    assert "stage-a" not in str(change.to_dict())


@pytest.mark.asyncio
async def test_agent_state_history_and_search_delegate_to_memory() -> None:
    memory = FakeMemory()
    store = AgentStateFacade(memory=memory).bind(key="demo", model=DemoState)

    assert await store.history(limit=3) == ["history"]
    assert await store.search("stage") == ["search"]
    assert memory.history_calls[0][0] == "demo"
    assert memory.search_calls[0]["key"] == "demo"


def test_node_context_state_delegates_to_bound_facade() -> None:
    memory = FakeMemory()
    services = NodeServices(
        channels=SimpleNamespace(),
        continuation_store=SimpleNamespace(),
        artifact_store=SimpleNamespace(),
        memory_facade=memory,
        agent_state=AgentStateFacade(memory=memory),
    )
    context = NodeContext(
        run_id="run",
        session_id="session",
        graph_id="graph",
        node_id="node",
        services=services,
    )

    store = context.state("demo", model=DemoState)

    assert store.key == "demo"
    assert context.memory() is memory
