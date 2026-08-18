from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.services.continuations.canonical_store import (
    CanonicalContinuationLeaseStore,
    CanonicalContinuationStore,
    bind_canonical_continuation_lease_store,
    bind_canonical_continuation_store,
)
from aethergraph.services.continuations.continuation import (
    ContinuationDraft,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
)
from aethergraph.storage.contracts import (
    ContinuationDraft as CanonicalDraft,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalContinuationLeaseRepository,
    LocalContinuationRepository,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 22, tzinfo=UTC)
SCHEDULED = NOW + timedelta(minutes=1)
SECRET = b"canonical-projection-test-secret-32-bytes"
OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _draft(
    continuation_id: str = "cont-1",
    *,
    run_id: str = "run-1",
    node_id: str = "node-1",
) -> ContinuationDraft:
    return ContinuationDraft(
        continuation_id=continuation_id,
        run_id=run_id,
        node_id=node_id,
        kind="approval",
        prompt="Approve?",
        resume_schema=None,
        deadline=NOW + timedelta(hours=1),
        poll={"interval_sec": 60},
        next_wakeup_at=SCHEDULED,
        attempts=3,
        channel="ui:session",
        created_at=NOW,
        payload=None,
        session_id="session-1",
        agent_id="agent-1",
        app_id="legacy-app",
        graph_id="graph-1",
        correlators=(Correlator("interaction", "public", message="interaction-1"),),
    )


@pytest.mark.asyncio
async def test_continuation_projection_is_lossless_scoped_and_revisioned(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    store = CanonicalContinuationStore(repository=repository, owner_scope=OWNER)

    created = await store.create(_draft())

    assert created.record.attempts == 3
    assert created.record.resume_schema is None
    assert created.record.payload is None
    assert created.record.poll == {"interval_sec": 60}
    assert created.record.app_id == "legacy-app"
    assert await store.get("run-1", "node-1") == created.record
    assert await store.get_by_id("run-1", "node-1", "cont-1") == created.record
    assert await store.resolve_token(created.token) == created.record
    other = CanonicalContinuationStore(
        repository=repository,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="other"),
    )
    assert await other.resolve_token(created.token) is None

    changed = replace(
        created.record,
        revision=2,
        resume_schema={"type": "object"},
        payload={"answer": 42},
        attempts=4,
    )
    stored = await store.update(changed, expected_revision=1)
    assert stored.resume_schema == {"type": "object"}
    assert stored.payload == {"answer": 42}
    with pytest.raises(StorageConflictError):
        await store.update(replace(stored, revision=3), expected_revision=1)

    correlated = await store.bind_correlator(
        continuation=stored,
        corr=Correlator("channel", "ui:session", message="message-1"),
    )
    page = await store.query(
        ContinuationQuery(
            session_id="session-1",
            correlator=Correlator("channel", "ui:session", message="message-1"),
            limit=2,
        )
    )
    assert page.items == (correlated,)
    closed = await store.close(
        correlated,
        status=ContinuationStatus.RESUMED,
        closed_at=NOW + timedelta(minutes=2),
    )
    assert closed.status is ContinuationStatus.RESUMED
    assert (
        await store.close(
            closed,
            status=ContinuationStatus.RESUMED,
            closed_at=NOW + timedelta(minutes=3),
        )
        == closed
    )
    await database.close()


@pytest.mark.asyncio
async def test_continuation_projection_rejects_ambiguous_run_node_identity(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-1",
        node_id="node-1",
    )
    for continuation_id in ("cont-1", "cont-2"):
        await repository.create(
            CanonicalDraft(
                continuation_id=continuation_id,
                kind="approval",
                scope=scope,
                created_at=NOW,
            )
        )
    store = CanonicalContinuationStore(repository=repository, owner_scope=OWNER)

    with pytest.raises(StorageIntegrityError, match="Multiple continuations"):
        await store.get("run-1", "node-1")
    await database.close()


@pytest.mark.asyncio
async def test_lease_projection_preserves_retry_reclaim_and_terminal_receipts(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    continuations = CanonicalContinuationStore(
        repository=LocalContinuationRepository(database=database, token_secret=SECRET),
        owner_scope=OWNER,
    )
    leases = CanonicalContinuationLeaseStore(
        repository=LocalContinuationLeaseRepository(database=database),
        owner_scope=OWNER,
    )
    await continuations.create(_draft())
    request = {
        "fire_id": "fire-1",
        "continuation_id": "cont-1",
        "run_id": "run-1",
        "node_id": "node-1",
        "scheduled_for": SCHEDULED,
        "worker_id": "worker-1",
        "now": SCHEDULED,
        "lease_until": SCHEDULED + timedelta(seconds=30),
    }

    first = await leases.claim(**request)
    assert first is not None and first.reclaimed is False
    assert await leases.claim(**{**request, "worker_id": "worker-2"}) is None
    retry_at = SCHEDULED + timedelta(minutes=1)
    assert await leases.record_failure(
        first,
        now=SCHEDULED + timedelta(seconds=1),
        next_attempt_at=retry_at,
        error="scheduler unavailable",
        dead_letter=False,
    )
    assert (
        await leases.claim(
            **{
                **request,
                "worker_id": "worker-2",
                "now": retry_at - timedelta(seconds=1),
                "lease_until": retry_at + timedelta(seconds=29),
            }
        )
        is None
    )
    retried = await leases.claim(
        **{
            **request,
            "worker_id": "worker-2",
            "now": retry_at,
            "lease_until": retry_at + timedelta(seconds=30),
        }
    )
    assert retried is not None and retried.reclaimed is False
    assert await leases.complete(retried, now=retry_at + timedelta(seconds=1))
    assert not await leases.complete(retried, now=retry_at + timedelta(seconds=2))
    receipt = await leases.get("run-1", "node-1", "fire-1")
    assert receipt is not None and receipt.status == "delivered"
    assert receipt.reclaimed is False
    assert await leases.get("other-run", "node-1", "fire-1") is None

    await continuations.create(_draft("cont-2", run_id="run-2", node_id="node-2"))
    stale_request = {
        **request,
        "fire_id": "fire-2",
        "continuation_id": "cont-2",
        "run_id": "run-2",
        "node_id": "node-2",
    }
    abandoned = await leases.claim(**stale_request)
    assert abandoned is not None
    reclaimed = await leases.claim(
        **{
            **stale_request,
            "worker_id": "recovery-worker",
            "now": stale_request["lease_until"] + timedelta(seconds=1),
            "lease_until": stale_request["lease_until"] + timedelta(seconds=31),
        }
    )
    assert reclaimed is not None and reclaimed.reclaimed is True
    assert await leases.record_failure(
        reclaimed,
        now=stale_request["lease_until"] + timedelta(seconds=2),
        next_attempt_at=None,
        error="exhausted",
        dead_letter=True,
    )
    dead = await leases.get("run-2", "node-2", "fire-2")
    assert dead is not None and dead.status == "dead_letter"
    await database.close()


def test_bundle_bindings_are_exact_and_public_docstrings_follow_required_format() -> None:
    continuation_repository = object()
    lease_repository = object()
    bundle = SimpleNamespace(
        continuations=continuation_repository,
        continuation_leases=lease_repository,
    )
    continuation_store = bind_canonical_continuation_store(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=OWNER,
    )
    lease_store = bind_canonical_continuation_lease_store(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=OWNER,
    )
    assert continuation_store._repository is continuation_repository
    assert lease_store._repository is lease_repository
    assert callable(continuation_store.create) and callable(continuation_store.query)
    assert callable(lease_store.claim) and callable(lease_store.complete)

    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    owners = (
        CanonicalContinuationStore,
        CanonicalContinuationLeaseStore,
    )
    members = [
        member
        for owner in owners
        for name, member in inspect.getmembers(owner, inspect.isfunction)
        if not name.startswith("_")
    ]
    members.extend((bind_canonical_continuation_store, bind_canonical_continuation_lease_store))
    for member in members:
        docstring = inspect.getdoc(member) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), member.__qualname__
        assert positions == tuple(sorted(positions)), member.__qualname__
        assert docstring.count("```python") >= 2, member.__qualname__
