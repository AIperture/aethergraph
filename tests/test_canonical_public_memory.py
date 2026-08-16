from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.agent_state import AgentStateFacade
from aethergraph.services.memory import (
    CanonicalMemoryFacadeFactory,
    CanonicalPublicMemoryFacade,
)
from aethergraph.services.memory.canonical_prompt import CanonicalPromptMemoryMixin
from aethergraph.storage.contracts import (
    EventQuery,
    SearchDocument,
    SearchMode,
    StorageIntegrityError,
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


class _LLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat(self, messages: list[dict], **kwargs):
        self.calls.append((messages, kwargs))
        return self.response, {"input_tokens": 10, "output_tokens": 5}


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
        with pytest.raises(ValueError, match="chat role"):
            await memory.append_chat_turn("visitor", "hello")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tags"):
            await memory.append_chat_turn("user", "hello", tags="chat")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="kinds"):
            await memory.query_events(kinds="chat.turn")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tags"):
            await memory.query_events(tags="chat")  # type: ignore[arg-type]
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


@pytest.mark.asyncio
async def test_public_memory_prompt_segments_are_canonical_and_chronological(
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
        StorageScope(session_id="session-1", run_id="run-1"),
        logical_scope_id="session:session-1",
    )
    try:
        await memory.record_chat_user("first", tags=["session.chat"])
        await memory.append_chat_turn("assistant", "second", tags=["session.chat"])
        await memory.append_event(
            kind="tool_result",
            data={"tool": "search"},
            tool="search",
            text="three hits",
            inputs=[{"query": "first"}],
            outputs=[{"count": 3}],
            tags=["verified"],
        )
        assert await memory.recent_chat() == [
            {"role": "user", "text": "first"},
            {"role": "assistant", "text": "second"},
        ]

        summary = await memory.distill_long_term(
            include_kinds=["chat.turn"],
            include_tags=["chat"],
            use_llm=False,
        )
        segments = await memory.build_prompt_segments(
            recent_chat_limit=10,
            include_recent_tools=True,
            tool="search",
            tool_limit=5,
            use_persistence=True,
        )

        assert summary["source_event_ids"] == ["event-1", "event-2"]
        assert summary["text"] == "[user] first\n[assistant] second"
        tool_timestamp = segments["recent_tools"][0]["ts"]
        assert datetime.fromisoformat(tool_timestamp).tzinfo is not None
        assert segments == {
            "long_term": "[user] first\n[assistant] second",
            "recent_chat": [
                {"role": "user", "text": "first"},
                {"role": "assistant", "text": "second"},
            ],
            "recent_tools": [
                {
                    "ts": tool_timestamp,
                    "tool": "search",
                    "message": "three hits",
                    "inputs": [{"query": "first"}],
                    "outputs": [{"count": 3}],
                    "tags": ["verified"],
                }
            ],
        }
        stored = await memory.get_latest_summary()
        assert stored is not None
        assert stored["event_id"] == summary["event_id"]
        assert await memory.record_state("prompt-state", {"ready": True})
        assert await memory.get_latest_state("prompt-state") == {"ready": True}
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_prompt_failures_do_not_become_empty_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-1"),
        logical_scope_id="run:run-1",
    )
    try:

        async def fail_query(_query):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(memory.canonical, "durable_query", fail_query)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await memory.list_summaries()
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await memory.build_prompt_segments()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_llm_distillation_is_explicit_and_strict(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    factory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        clock=clock.now,
    )
    missing = factory.for_public_execution(
        StorageScope(run_id="run-missing"),
        logical_scope_id="run:run-missing",
    )
    malformed_llm = _LLM("not-json")
    malformed = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        llm=malformed_llm,
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-malformed"),
        logical_scope_id="run:run-malformed",
    )
    strict_llm = _LLM(
        json.dumps(
            {
                "summary": "A concise summary.",
                "key_facts": ["one"],
                "open_loops": [],
            }
        )
    )
    strict = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        llm=strict_llm,
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-strict"),
        logical_scope_id="run:run-strict",
    )
    try:
        await missing.append_chat_turn("user", "missing client")
        await malformed.append_chat_turn("user", "malformed output")
        await strict.append_chat_turn("user", "strict output")

        with pytest.raises(RuntimeError, match="LLM client not configured"):
            await missing.distill_summary(use_llm=True)
        with pytest.raises(json.JSONDecodeError):
            await malformed.distill_summary(use_llm=True)
        summary = await strict.distill_summary(use_llm=True)

        assert summary["summary"] == "A concise summary."
        assert strict_llm.calls[0][1] == {"output_format": "json"}
        assert await malformed.list_summaries() == []
        assert (await strict.list_summaries())[0]["event_id"] == summary["event_id"]
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_prompt_bounds_and_metadata_conflicts_fail_directly(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        clock=clock.now,
    ).for_public_execution(
        StorageScope(run_id="run-1"),
        logical_scope_id="run:run-1",
    )
    try:
        await memory.append_chat_turn("user", "hello")
        assert await memory.recent_chat(limit=0) == []
        with pytest.raises(TypeError, match="roles"):
            await memory.recent_chat(roles="user")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tags"):
            await memory.recent_chat(tags="chat")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="level"):
            await memory.query_events(level="application")
        with pytest.raises(ValueError, match="level"):
            await memory.recent_events(level="application")
        with pytest.raises(ValueError, match="max_events"):
            await memory.distill_summary(max_events=0)
        with pytest.raises(ValueError, match="conflicts"):
            await memory.distill_summary(extra_data={"num_events": 99})
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_memory_search_hydrates_exact_rank_and_rejects_stale_projection(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(project_id="project-1"),
        clock=clock.now,
        event_id_factory=_Identities(),
    ).for_public_execution(
        StorageScope(run_id="run-1"),
        logical_scope_id="run:run-1",
        deprecated_app_id="app-legacy",
    )
    try:
        first = await memory.append_chat_turn(
            "user",
            "canonical migration evidence",
            tags=["verified"],
        )
        await memory.append_chat_turn("assistant", "unrelated response")

        hits = await memory.search_events(
            query="migration",
            mode=SearchMode.LEXICAL,
            tags=["chat", "verified"],
        )

        assert len(hits) == 1
        assert hits[0].event == first
        assert hits[0].event.app_id == "app-legacy"
        assert hits[0].score > 0
        assert hits[0].mode is SearchMode.LEXICAL

        record = await memory.canonical.get_event(first.event_id)
        assert record is not None
        await bundle.search.upsert(
            SearchDocument(
                corpus="memory",
                item_id=first.event_id,
                text="stale projection",
                scope=memory.scope,
                occurred_at=record.occurred_at,
                tags=("chat", "verified"),
                metadata={
                    "event_cursor": "stale-cursor",
                    "kind": "chat.turn",
                    "tags": ["chat", "verified"],
                    "stage": "user",
                },
            )
        )
        with pytest.raises(StorageIntegrityError, match="stale"):
            await memory.search_events(
                query="stale",
                mode=SearchMode.LEXICAL,
            )
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
        CanonicalPublicMemoryFacade.recent_chat,
        CanonicalPublicMemoryFacade.list_summaries,
        CanonicalPublicMemoryFacade.get_latest_summary,
        CanonicalPublicMemoryFacade.distill_summary,
        CanonicalPublicMemoryFacade.build_prompt_segments,
        CanonicalPublicMemoryFacade.record_state,
        CanonicalPublicMemoryFacade.record_chat_user,
        CanonicalPublicMemoryFacade.distill_long_term,
        CanonicalPublicMemoryFacade.search_events,
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
    prompt_source = inspect.getsource(CanonicalPromptMemoryMixin)
    assert "except Exception" not in prompt_source
    assert "getattr(" not in prompt_source
    assert "ScopedIndices" not in prompt_source
