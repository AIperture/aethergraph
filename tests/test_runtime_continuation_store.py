from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect

import pytest

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.services.continuations.continuation import (
    ContinuationDraft,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
)
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore

NOW = datetime(2026, 8, 15, 22, tzinfo=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "filesystem"])
async def test_runtime_store_create_is_atomic_tokenless_and_revisioned(tmp_path, kind) -> None:
    store = (
        InMemoryContinuationStore(secret=b"runtime-secret")
        if kind == "memory"
        else FSContinuationStore(tmp_path / "continuations", secret=b"runtime-secret")
    )
    draft = ContinuationDraft(
        continuation_id="cont-1",
        run_id="run-1",
        node_id="node-1",
        kind="choice",
        created_at=NOW,
        session_id="session-1",
        app_id="legacy-app",
        correlators=(Correlator("interaction", "public", message="interaction-1"),),
    )

    created = await store.create(draft)

    assert created.record.revision == 1
    assert "token" not in created.record.to_dict()
    assert await store.resolve_token(created.token) == created.record
    assert created.record.to_dict()["metadata"]["compatibility_metadata"]["app_id"] == {
        "value": "legacy-app",
        "deprecated": True,
        "scheduled_removal": "future breaking release",
    }
    changed = replace(
        created.record,
        revision=2,
        next_wakeup_at=NOW + timedelta(minutes=1),
    )
    stored = await store.update(changed, expected_revision=1)
    with pytest.raises(RuntimeError, match="revision conflict"):
        await store.update(replace(stored, revision=3), expected_revision=1)
    closed = await store.close(
        stored,
        status=ContinuationStatus.RESUMED,
        closed_at=NOW + timedelta(minutes=2),
    )
    assert closed.status is ContinuationStatus.RESUMED
    assert await store.get("run-1", "node-1") == closed
    assert not any(
        hasattr(store, name)
        for name in ("mint_token", "save", "delete", "get_by_token", "list_waits")
    )

    if kind == "filesystem":
        raw = (tmp_path / "continuations" / "continuations.v2.json").read_text("utf-8")
        assert created.token not in raw
        assert "legacy-app" in raw
        reopened = FSContinuationStore(tmp_path / "continuations", secret=b"runtime-secret")
        assert await reopened.get_by_id("cont-1") == closed


@pytest.mark.asyncio
async def test_runtime_queries_are_bounded_index_explicit_and_open_aware() -> None:
    store = InMemoryContinuationStore(secret=b"runtime-secret")
    expired = (
        await store.create(
            ContinuationDraft(
                continuation_id="expired",
                run_id="run-expired",
                node_id="node",
                kind="user_input",
                created_at=NOW,
                deadline=NOW - timedelta(seconds=1),
                next_wakeup_at=NOW - timedelta(seconds=1),
                session_id="session-1",
            )
        )
    ).record
    open_wait = (
        await store.create(
            ContinuationDraft(
                continuation_id="open",
                run_id="run-open",
                node_id="node",
                kind="user_input",
                created_at=NOW + timedelta(seconds=1),
                deadline=NOW + timedelta(minutes=1),
                next_wakeup_at=NOW + timedelta(minutes=1),
                session_id="session-1",
            )
        )
    ).record

    page = await store.query(
        ContinuationQuery(
            session_id="session-1",
            kinds=("user_input",),
            open_at=NOW,
            limit=2,
        )
    )
    due = await store.query(ContinuationQuery(due_at_or_before=NOW, limit=1))

    assert page.items == (open_wait,)
    assert due.items == (expired,)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        ContinuationQuery(limit=1001)


def test_runtime_continuation_public_docstrings_follow_required_format() -> None:
    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    for owner in (
        AsyncContinuationStore,
        InMemoryContinuationStore,
        FSContinuationStore,
    ):
        for name, member in inspect.getmembers(owner, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (owner.__name__, name)
            assert positions == tuple(sorted(positions)), (owner.__name__, name)
            assert docstring.count("```python") >= 2, (owner.__name__, name)
