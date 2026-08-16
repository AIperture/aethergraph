from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

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


def test_public_memory_docstrings_and_surface_are_explicit() -> None:
    for member in (
        CanonicalPublicMemoryFacade.__init__,
        CanonicalPublicMemoryFacade.append_event,
        CanonicalPublicMemoryFacade.append_chat_turn,
        CanonicalPublicMemoryFacade.get_event,
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
