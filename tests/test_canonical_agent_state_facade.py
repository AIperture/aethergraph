from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.agent_state import (
    AgentStateConflictError,
    CanonicalAgentStateFacade,
    CanonicalAgentStateHandle,
    project_agent_state_scope,
)
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.contracts import (
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://tests/agent-state"
_SECRET = b"canonical-agent-state-secret-32-bytes"


@dataclass
class _AgentState:
    count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"count": self.count}

    @classmethod
    def from_dict(cls, value: dict | None) -> _AgentState:
        return cls(count=int((value or {}).get("count") or 0))


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 8, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


def _open_bundle(root: Path):
    provider = LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    )
    return provider.open(
        StorageOpenRequest(
            workspace_id="agent-state-tests",
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


def _base_scope() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
        agent_id="agent-1",
    )


def test_canonical_agent_state_scope_projection_is_exact_and_app_free() -> None:
    scope = _base_scope()

    assert project_agent_state_scope(scope, level="session").as_filter() == {
        "tenant_id": "tenant-1",
        "project_id": "project-1",
        "org_id": "org-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "agent_id": "agent-1",
    }
    assert project_agent_state_scope(scope, level="run").as_filter() == {
        "tenant_id": "tenant-1",
        "project_id": "project-1",
        "org_id": "org-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "graph_id": "graph-1",
        "agent_id": "agent-1",
    }
    with pytest.raises(ValueError, match="scope level"):
        project_agent_state_scope(scope, level="invalid")  # type: ignore[arg-type]


def test_canonical_agent_state_handle_cache_normalizes_metadata_order() -> None:
    facade = CanonicalAgentStateFacade(state_store=object(), scope=_base_scope())  # type: ignore[arg-type]

    first = facade.bind(key="planner", meta={"a": 1, "b": 2})
    second = facade.bind(key="planner", meta={"b": 2, "a": 1})

    assert first is second


def test_explicit_runtime_scope_can_omit_agent_for_shared_session_state() -> None:
    facade = CanonicalAgentStateFacade(state_store=object(), scope=_base_scope())  # type: ignore[arg-type]
    shared = Scope(
        org_id="org-1",
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
    )

    handle = facade.bind(key="session-envelope", level="session", scope=shared)

    assert handle.scope.as_filter() == {
        "tenant_id": "tenant-1",
        "project_id": "project-1",
        "org_id": "org-1",
        "user_id": "user-1",
        "session_id": "session-1",
    }
    assert facade.bind(key="session-envelope", level="session") is not handle


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("org_id", "other-org"),
        ("user_id", "other-user"),
        ("session_id", "other-session"),
        ("run_id", "other-run"),
        ("graph_id", "other-graph"),
        ("agent_id", "other-agent"),
    ),
)
def test_explicit_runtime_scope_cannot_switch_trusted_identity(
    field: str,
    value: str,
) -> None:
    values = {
        "org_id": "org-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "run_id": "run-1",
        "graph_id": "graph-1",
        "agent_id": "agent-1",
    }
    values[field] = value
    facade = CanonicalAgentStateFacade(state_store=object(), scope=_base_scope())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=field):
        facade.bind(key="session-envelope", level="session", scope=Scope(**values))


@pytest.mark.asyncio
async def test_canonical_agent_state_round_trip_and_history(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    handle = CanonicalAgentStateFacade(state_store=bundle.state, scope=_base_scope()).bind(
        key="planner",
        model=_AgentState,
        default_factory=_AgentState,
        tags=("planner",),
    )
    try:
        assert await handle.load() == _AgentState()
        first = await handle.commit(
            _AgentState(count=1),
            reason="planned",
            expected_revision=0,
        )
        second = await handle.commit(
            _AgentState(count=2),
            reason="executed",
            expected_revision=1,
        )

        assert first is not None and first.revision == 1
        assert second is not None and second.revision == 2
        assert await handle.load(force=True) == _AgentState(count=2)
        history = await handle.history(limit=10)
        assert [record.revision for record in history.items] == [2, 1]
        assert history.items[0].metadata["reason"] == "executed"
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_agent_state_cross_handle_cas_fails_without_retry(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    first = CanonicalAgentStateFacade(state_store=bundle.state, scope=_base_scope()).bind(
        key="planner", model=_AgentState
    )
    second = CanonicalAgentStateFacade(state_store=bundle.state, scope=_base_scope()).bind(
        key="planner", model=_AgentState
    )
    try:
        await first.load()
        await second.load()
        await first.commit(_AgentState(count=1), expected_revision=0)

        with pytest.raises(AgentStateConflictError) as conflict:
            await second.commit(_AgentState(count=2), expected_revision=0)

        assert conflict.value.expected_revision == 0
        assert conflict.value.actual_revision == 1
        assert await second.load(force=True) == _AgentState(count=1)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_agent_state_local_backend_never_writes_provider(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    facade = CanonicalAgentStateFacade(state_store=bundle.state, scope=_base_scope())
    handle = facade.bind(key="scratch", model=_AgentState, backend="local")
    try:
        await handle.commit(_AgentState(count=3), expected_revision=0)

        assert await handle.load(force=True) == _AgentState(count=3)
        assert (
            await bundle.state.get(
                project_agent_state_scope(_base_scope(), level=None),
                "agent_state:state.snapshot",
                "scratch",
            )
            is None
        )
        assert (await handle.history()).items == ()
    finally:
        await bundle.close()


def test_agent_state_public_docstrings_and_probe_removal_are_locked() -> None:
    for name in ("load", "commit", "update", "history"):
        docstring = inspect.getdoc(getattr(CanonicalAgentStateHandle, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2
    bind_docstring = inspect.getdoc(CanonicalAgentStateFacade.bind) or ""
    assert bind_docstring.count("```python") >= 2

    canonical_source = inspect.getsource(CanonicalAgentStateHandle)
    for method_name in (
        "get_latest_state_record",
        "append_state_snapshot",
        "append_external_resource_change",
        "list_state_history",
    ):
        assert method_name not in canonical_source
    assert "_call_memory_method" not in canonical_source
