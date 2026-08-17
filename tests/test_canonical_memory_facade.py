from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.memory import CanonicalMemoryFacade
from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    PageRequest,
    SearchMode,
    SearchProjectionStatus,
    SortDirection,
    StorageCapabilityError,
    StorageConflictError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://tests/memory"
_SECRET = b"canonical-memory-secret-material-32b"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 9, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


class _Monotonic:
    value = 0.0

    def __call__(self) -> float:
        return self.value


class _FailingSearch:
    async def upsert_many(self, documents):
        raise RuntimeError("index unavailable")


def _scope() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        agent_id="agent-1",
    )


def _open_bundle(root: Path):
    provider = LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    )
    return provider.open(
        StorageOpenRequest(
            workspace_id="memory-tests",
            workspace_root=root.resolve(),
            owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=_Clock(),
            secrets=_Secrets(),
        )
    )


def _draft(event_id: str, second: int, text: str) -> EventDraft:
    return EventDraft(
        event_id=event_id,
        occurred_at=datetime(2026, 8, 16, 9, 0, second, tzinfo=UTC),
        scope=_scope(),
        kind="chat.message",
        stage="assistant",
        topic="conversation",
        text=text,
        tags=("chat",),
        payload={"role": "assistant"},
    )


@pytest.mark.asyncio
async def test_canonical_memory_commit_cache_search_and_retry_are_coherent(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
    )
    try:
        receipt = await memory.append_event(
            event_id="event-1",
            occurred_at=datetime(2026, 8, 16, 9, 0, 1, tzinfo=UTC),
            kind="chat.message",
            stage="assistant",
            topic="conversation",
            text="hello aethergraph",
            tags=("chat",),
            payload={"role": "assistant"},
        )
        retried = await memory.append_many((_draft("event-1", 1, "hello aethergraph"),))

        assert receipt.events == retried.events
        assert receipt.event_cursor == receipt.events[0].cursor
        assert receipt.indexed_cursor is not None
        assert retried.indexed_cursor == receipt.indexed_cursor
        assert [event.event_id for event in await memory.recent_hot()] == ["event-1"]
        hits = await memory.search(
            query="aethergraph",
            mode=SearchMode.LEXICAL,
            require_indexed_cursor=receipt.indexed_cursor,
        )
        assert [hit.item_id for hit in hits] == ["event-1"]
        assert await memory.indexed_cursor() == receipt.indexed_cursor
        assert (
            await memory.wait_until_indexed(receipt.indexed_cursor, 0.0) == receipt.indexed_cursor
        )
        with pytest.raises(StorageCapabilityError, match="semantic"):
            await memory.search(query="aethergraph", mode=SearchMode.SEMANTIC)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_memory_durable_query_is_cursor_paged_and_hot_is_bounded(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
        hot_max_events=1,
    )
    try:
        await memory.append_many(
            (
                _draft("event-1", 1, "first"),
                _draft("event-2", 2, "second"),
            )
        )
        query = EventQuery(
            scope=_scope(),
            page=PageRequest(limit=1),
            kinds=("chat.message",),
            order=SortDirection.ASCENDING,
        )
        first_page = await memory.durable_query(query)
        second_page = await memory.durable_query(
            replace(query, page=PageRequest(limit=1, cursor=first_page.next_cursor))
        )

        assert [event.event_id for event in first_page.items] == ["event-1"]
        assert [event.event_id for event in second_page.items] == ["event-2"]
        assert [event.event_id for event in await memory.recent_hot()] == ["event-2"]
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_memory_hot_ttl_never_falls_back_to_durable(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    clock = _Monotonic()
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
        hot_ttl_seconds=5.0,
        monotonic_clock=clock,
    )
    try:
        await memory.append_many((_draft("event-1", 1, "expires"),))
        clock.value = 6.0

        assert await memory.recent_hot() == ()
        durable = await memory.durable_query(EventQuery(scope=_scope()))
        assert [event.event_id for event in durable.items] == ["event-1"]
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_memory_search_failure_is_visible_after_authoritative_commit(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=_FailingSearch(),  # type: ignore[arg-type]
        scope=_scope(),
    )
    try:
        receipt = await memory.append_many((_draft("event-1", 1, "durable"),))

        assert receipt.projection_status == "failed"
        assert receipt.projection_diagnostic == "RuntimeError: search projection failed"
        durable = await bundle.memory_events.get(_scope(), "event-1")
        assert durable is not None
        assert [event.event_id for event in await memory.recent_hot()] == ["event-1"]
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_failed_projection_intent_survives_reopen_and_retries_exact_event(
    tmp_path: Path,
) -> None:
    first_bundle = _open_bundle(tmp_path)
    first = CanonicalMemoryFacade(
        event_store=first_bundle.memory_events,
        state_store=first_bundle.state,
        search_backend=_FailingSearch(),  # type: ignore[arg-type]
        scope=_scope(),
    )
    draft = _draft("event-retry", 1, "durable retry")
    failed = await first.append_many((draft,))
    intent = await first.get_search_projection_intent("event-retry")
    assert failed.projection_status == "failed"
    assert failed.created == (True,)
    assert intent is not None
    assert intent.status is SearchProjectionStatus.FAILED
    assert intent.attempts == 1
    await first_bundle.close()

    second_bundle = _open_bundle(tmp_path)
    second = CanonicalMemoryFacade(
        event_store=second_bundle.memory_events,
        state_store=second_bundle.state,
        search_backend=second_bundle.search,
        scope=_scope(),
    )
    try:
        retried = await second.append_many((draft,))
        restored = await second.get_search_projection_intent("event-retry")
        durable = await second.durable_query(EventQuery(scope=_scope()))

        assert retried.projection_status == "indexed"
        assert retried.created == (False,)
        assert restored is not None
        assert restored.status is SearchProjectionStatus.INDEXED
        assert restored.attempts == 2
        assert [event.event_id for event in durable.items] == ["event-retry"]
    finally:
        await second_bundle.close()


@pytest.mark.asyncio
async def test_canonical_memory_concurrent_append_has_no_loss_or_hot_duplicates(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
    )
    try:
        await asyncio.gather(
            *(
                memory.append_event(
                    event_id=f"event-{index}",
                    occurred_at=datetime(2026, 8, 16, 9, 1, index, tzinfo=UTC),
                    kind="memory.event",
                    text=f"event {index}",
                )
                for index in range(20)
            )
        )

        hot_ids = [event.event_id for event in await memory.recent_hot(limit=20)]
        durable = await memory.durable_query(EventQuery(scope=_scope(), page=PageRequest(limit=20)))
        assert len(hot_ids) == len(set(hot_ids)) == 20
        assert len(durable.items) == 20
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_memory_state_uses_one_atomic_state_authority(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    memory = CanonicalMemoryFacade(
        event_store=bundle.memory_events,
        state_store=bundle.state,
        search_backend=bundle.search,
        scope=_scope(),
    )
    try:
        first = await memory.commit_state(
            key="agent:writer",
            value={"draft": 1},
            expected_revision=0,
            metadata={"source": "test"},
        )
        second = await memory.commit_state(
            key="agent:writer",
            value={"draft": 2},
            expected_revision=1,
        )

        assert first.revision == 1
        assert second.revision == 2
        assert await memory.current_state(key="agent:writer") == second
        oldest = await memory.state_history(
            key="agent:writer",
            limit=1,
            order=SortDirection.ASCENDING,
        )
        newest = await memory.state_history(key="agent:writer", limit=1)
        assert [record.revision for record in oldest.items] == [1]
        assert [record.revision for record in newest.items] == [2]
        assert oldest.next_cursor is not None

        with pytest.raises(StorageConflictError):
            await memory.commit_state(
                key="agent:writer",
                value={"draft": 3},
                expected_revision=1,
            )
        with pytest.raises(ValueError, match="key"):
            await memory.current_state(key=" agent:writer")
        with pytest.raises(ValueError, match="kind"):
            await memory.current_state(key="agent:writer", kind="state.snapshot ")

        event_page = await memory.durable_query(EventQuery(scope=_scope()))
        assert event_page.items == ()
        commit_source = inspect.getsource(CanonicalMemoryFacade.commit_state)
        assert "_events" not in commit_source
        assert "append(" not in commit_source
    finally:
        await bundle.close()


def test_canonical_memory_surface_has_no_legacy_identity_or_payload_aliases() -> None:
    event_fields = {item.name for item in fields(EventDraft)}
    assert {"app_id", "client_id", "tool", "embedding", "inputs", "outputs"}.isdisjoint(
        event_fields
    )

    for name in (
        "append_event",
        "append_many",
        "durable_query",
        "get_event",
        "commit_state",
        "current_state",
        "state_history",
        "recent_hot",
        "search",
        "indexed_cursor",
        "wait_until_indexed",
    ):
        docstring = inspect.getdoc(getattr(CanonicalMemoryFacade, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2
