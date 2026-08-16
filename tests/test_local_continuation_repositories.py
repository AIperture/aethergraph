from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    ContinuationCorrelator,
    ContinuationDraft,
    ContinuationLeaseQuery,
    ContinuationLeaseRequest,
    ContinuationLeaseStatus,
    ContinuationQuery,
    ContinuationStatus,
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalContinuationLeaseRepository,
    LocalContinuationRepository,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 21, tzinfo=UTC)
SECRET = b"local-continuation-test-secret-32-bytes-minimum"


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _scope(*, project_id: str = "project-1") -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id=project_id,
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
    )


def _draft(
    continuation_id: str,
    *,
    created_at: datetime = NOW,
    next_wakeup_at: datetime | None = None,
    correlators: tuple[ContinuationCorrelator, ...] = (),
) -> ContinuationDraft:
    return ContinuationDraft(
        continuation_id=continuation_id,
        kind="approval",
        scope=_scope(),
        created_at=created_at,
        prompt="Approve?",
        resume_schema={"type": "object"},
        payload={"answer": 42},
        poll_payload={"attempt": 0},
        metadata={"service_context": {"interaction_id": "interaction-1"}},
        deadline=created_at + timedelta(hours=1),
        next_wakeup_at=next_wakeup_at,
        channel="ui:session",
        correlators=correlators,
    )


def _claim_request(
    continuation_id: str,
    *,
    fire_id: str = "fire-1",
    worker_id: str = "worker-1",
    now: datetime = NOW + timedelta(minutes=1),
) -> ContinuationLeaseRequest:
    return ContinuationLeaseRequest(
        fire_id=fire_id,
        continuation_id=continuation_id,
        scope=_scope(),
        scheduled_for=NOW + timedelta(seconds=30),
        worker_id=worker_id,
        now=now,
        lease_until=now + timedelta(seconds=30),
    )


@pytest.mark.asyncio
async def test_continuation_create_token_scope_and_atomic_correlators(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    correlator = ContinuationCorrelator(
        scheme="channel", channel="ui:session", thread="thread-1", message="message-1"
    )
    created = await repository.create(_draft("cont-1", correlators=(correlator,)))

    assert created.record.revision == 1
    assert created.record.token_digest.startswith("hmac-sha256:")
    assert created.token not in created.record.token_digest
    assert await repository.get(_scope(), "cont-1") == created.record
    assert created.record.metadata["service_context"]["interaction_id"] == "interaction-1"
    assert await repository.get(StorageScope(project_id="other"), "cont-1") is None
    assert await repository.get(StorageScope(), "cont-1") is None
    assert await repository.resolve_token(created.token) == created.record
    assert await repository.resolve_token("unknown-token") is None
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.create(_draft("cont-1"))

    by_correlator = await repository.query(ContinuationQuery(scope=_scope(), correlator=correlator))
    assert by_correlator.items == (created.record,)
    raw_rows = await database.fetch_all(
        "SELECT * FROM local_continuations WHERE continuation_id = ?", ("cont-1",)
    )
    assert created.token not in " ".join(str(value) for value in raw_rows[0])
    await database.close()


@pytest.mark.asyncio
async def test_continuation_cas_and_correlator_binding_are_atomic(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    current = (await repository.create(_draft("cont-1"))).record
    correlator = ContinuationCorrelator(scheme="channel", channel="ui:session")

    bound = await repository.bind_correlator(
        current.scope, current.continuation_id, correlator, current.revision
    )
    assert bound.revision == 2
    assert bound.correlators == (correlator,)
    assert (
        await repository.bind_correlator(
            current.scope, current.continuation_id, correlator, current.revision
        )
        == bound
    )
    other = ContinuationCorrelator(scheme="channel", channel="other")
    with pytest.raises(StorageConflictError):
        await repository.bind_correlator(
            current.scope, current.continuation_id, other, current.revision
        )
    with pytest.raises(StorageNotFoundError):
        await repository.bind_correlator(
            StorageScope(project_id="other"), current.continuation_id, other, bound.revision
        )

    with pytest.raises(StorageIntegrityError, match="immutable"):
        await repository.compare_and_set(
            replace(bound, revision=3, token_digest="hmac-sha256:changed"), 2
        )
    resumed = replace(
        bound,
        revision=3,
        status=ContinuationStatus.RESUMED,
        closed_at=NOW + timedelta(minutes=2),
    )
    assert await repository.compare_and_set(resumed, 2) == resumed
    with pytest.raises(StorageConflictError, match="Terminal"):
        await repository.compare_and_set(replace(resumed, revision=4), 3)
    await database.close()


@pytest.mark.asyncio
async def test_continuation_due_and_created_queries_have_bound_cursors(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    records = []
    for index in range(3):
        records.append(
            (
                await repository.create(
                    _draft(
                        f"cont-{index}",
                        created_at=NOW + timedelta(seconds=index),
                        next_wakeup_at=NOW + timedelta(minutes=1, seconds=index),
                    )
                )
            ).record
        )
    query = ContinuationQuery(scope=_scope(), page=PageRequest(limit=2))
    first = await repository.query(query)
    second = await repository.query(
        replace(query, page=PageRequest(limit=2, cursor=first.next_cursor))
    )
    assert (*first.items, *second.items) == tuple(reversed(records))
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query(
            replace(
                query,
                statuses=(ContinuationStatus.WAITING,),
                page=PageRequest(limit=2, cursor=first.next_cursor),
            )
        )
    due = await repository.query(
        ContinuationQuery(
            scope=_scope(),
            due_at_or_before=NOW + timedelta(minutes=1, seconds=1),
        )
    )
    assert due.items == tuple(records[:2])
    open_page = await repository.query(
        ContinuationQuery(
            scope=_scope(),
            open_at=NOW + timedelta(hours=2),
        )
    )
    assert open_page.items == ()
    await database.close()


@pytest.mark.asyncio
async def test_lease_claim_contention_retry_reclaim_and_terminal_receipt(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    continuations = LocalContinuationRepository(database=database, token_secret=SECRET)
    leases = LocalContinuationLeaseRepository(database=database)
    await continuations.create(_draft("cont-1", next_wakeup_at=NOW + timedelta(seconds=30)))
    request = _claim_request("cont-1")
    assert await leases.claim(replace(request, scheduled_for=NOW + timedelta(seconds=31))) is None

    claims = await asyncio.gather(*(leases.claim(request) for _ in range(20)))
    assert sum(claim is not None for claim in claims) == 1
    lease = next(claim for claim in claims if claim is not None)
    assert lease.attempts == lease.revision == 1
    regressed = request.now - timedelta(seconds=1)
    with pytest.raises(StorageIntegrityError, match="clock moved backward"):
        await leases.claim(
            replace(request, now=regressed, lease_until=regressed + timedelta(seconds=30))
        )
    retry_time = request.now + timedelta(minutes=1)
    retry = replace(
        lease,
        revision=2,
        status=ContinuationLeaseStatus.RETRY,
        updated_at=request.now + timedelta(seconds=1),
        worker_id=None,
        lease_until=None,
        next_attempt_at=retry_time,
        last_error="scheduler unavailable",
    )
    assert await leases.compare_and_set(retry, 1) == retry
    before_retry = retry_time - timedelta(seconds=1)
    assert (
        await leases.claim(
            replace(
                request,
                now=before_retry,
                lease_until=before_retry + timedelta(seconds=30),
            )
        )
        is None
    )
    reclaimed = await leases.claim(
        replace(
            request,
            worker_id="worker-2",
            now=retry_time,
            lease_until=retry_time + timedelta(seconds=30),
        )
    )
    assert reclaimed is not None
    assert reclaimed.attempts == 2 and reclaimed.revision == 3
    assert reclaimed.worker_id == "worker-2"
    delivered_at = retry_time + timedelta(seconds=1)
    delivered = replace(
        reclaimed,
        revision=4,
        status=ContinuationLeaseStatus.DELIVERED,
        updated_at=delivered_at,
        worker_id=None,
        lease_until=None,
        finished_at=delivered_at,
    )
    assert await leases.compare_and_set(delivered, 3) == delivered
    assert (
        await leases.claim(
            replace(
                request,
                worker_id="worker-3",
                now=delivered_at + timedelta(minutes=1),
                lease_until=delivered_at + timedelta(minutes=2),
            )
        )
        is None
    )
    with pytest.raises(StorageConflictError):
        await leases.compare_and_set(replace(delivered, revision=5), 4)
    await database.close()


@pytest.mark.asyncio
async def test_lease_identity_scope_renewal_and_cursor_query(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    continuations = LocalContinuationRepository(database=database, token_secret=SECRET)
    leases = LocalContinuationLeaseRepository(database=database)
    await continuations.create(_draft("cont-1", next_wakeup_at=NOW + timedelta(seconds=30)))
    await continuations.create(_draft("cont-2", next_wakeup_at=NOW + timedelta(seconds=30)))
    first = await leases.claim(_claim_request("cont-1", fire_id="fire-1"))
    second = await leases.claim(
        _claim_request("cont-2", fire_id="fire-2", now=NOW + timedelta(minutes=1, seconds=1))
    )
    assert first is not None and second is not None
    assert await leases.get(StorageScope(project_id="other"), first.fire_id) is None
    renewed = replace(
        first,
        revision=2,
        updated_at=first.updated_at + timedelta(seconds=1),
        lease_until=first.lease_until + timedelta(seconds=30),
    )
    assert await leases.compare_and_set(renewed, 1) == renewed
    with pytest.raises(StorageConflictError, match="worker"):
        await leases.compare_and_set(
            replace(
                renewed,
                revision=3,
                worker_id="other-worker",
                lease_until=renewed.lease_until + timedelta(seconds=30),
            ),
            2,
        )
    with pytest.raises(StorageIntegrityError, match="identity"):
        await leases.claim(replace(_claim_request("cont-2"), fire_id="fire-1"))

    query = ContinuationLeaseQuery(scope=_scope(), page=PageRequest(limit=1))
    page_one = await leases.query(query)
    page_two = await leases.query(
        replace(query, page=PageRequest(limit=1, cursor=page_one.next_cursor))
    )
    assert {record.fire_id for record in (*page_one.items, *page_two.items)} == {
        "fire-1",
        "fire-2",
    }
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await leases.query(
            replace(
                query,
                continuation_id="cont-1",
                page=PageRequest(limit=1, cursor=page_one.next_cursor),
            )
        )
    await database.close()


@pytest.mark.asyncio
async def test_continuation_schema_and_query_paths_are_canonical_and_indexed(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalContinuationRepository(database=database, token_secret=SECRET)
    LocalContinuationLeaseRepository(database=database)
    for table in (
        "local_continuations",
        "local_continuation_correlators",
        "local_continuation_leases",
    ):
        columns = {
            str(row["name"]) for row in await database.fetch_all(f"PRAGMA table_info({table})")
        }
        assert not {"app_id", "application_id", "client_id", "path", "token"} & columns
    due_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_continuations
            WHERE run_id = ? AND node_id = ? AND status = ?
              AND next_wakeup_at <= ?
            ORDER BY next_wakeup_at, continuation_id LIMIT ?
            """,
            ("run-1", "node-1", "waiting", NOW.isoformat(), 20),
        )
    )
    correlator_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT continuation_id
            FROM local_continuation_correlators
            WHERE scheme = ? AND channel = ? AND thread = ? AND message = ?
            LIMIT ?
            """,
            ("channel", "ui", "", "", 20),
        )
    )
    lease_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_continuation_leases
            WHERE run_id = ? AND node_id = ?
            ORDER BY updated_at DESC, fire_id DESC LIMIT ?
            """,
            ("run-1", "node-1", 20),
        )
    )
    assert "ix_local_continuations_scope_due" in due_plan
    assert "ix_local_continuation_correlators_lookup" in correlator_plan
    assert "ix_local_continuation_leases_scope_updated" in lease_plan
    assert "SCAN local_" not in f"{due_plan} {correlator_plan} {lease_plan}"
    indexes = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA index_list(local_continuations)")
    }
    assert "ix_local_continuations_session_open" in indexes
    await database.close()


@pytest.mark.asyncio
async def test_read_only_repositories_read_and_reject_mutation(tmp_path: Path) -> None:
    writable = _database(tmp_path, StorageOpenMode.READ_WRITE)
    continuations = LocalContinuationRepository(database=writable, token_secret=SECRET)
    leases = LocalContinuationLeaseRepository(database=writable)
    created = await continuations.create(
        _draft("cont-1", next_wakeup_at=NOW + timedelta(seconds=30))
    )
    claim = await leases.claim(_claim_request("cont-1"))
    assert claim is not None
    await writable.close()

    readonly = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly_continuations = LocalContinuationRepository(database=readonly, token_secret=SECRET)
    readonly_leases = LocalContinuationLeaseRepository(database=readonly)
    assert await readonly_continuations.resolve_token(created.token) == created.record
    assert await readonly_leases.get(_scope(), claim.fire_id) == claim
    with pytest.raises(StorageReadOnlyError):
        await readonly_continuations.create(_draft("new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly_continuations.bind_correlator(
            _scope(), "cont-1", ContinuationCorrelator(scheme="x", channel="y"), 1
        )
    with pytest.raises(StorageReadOnlyError):
        await readonly_leases.claim(_claim_request("cont-1", fire_id="new"))
    await readonly.close()


@pytest.mark.asyncio
async def test_persisted_continuation_corruption_is_typed(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalContinuationRepository(database=database, token_secret=SECRET)
    await repository.create(_draft("cont-1"))
    await database.execute(
        "UPDATE local_continuations SET payload_json = ? WHERE continuation_id = ?",
        ("[]", "cont-1"),
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await repository.get(_scope(), "cont-1")
    await database.close()


def test_local_continuation_repository_docstrings_follow_required_section_order() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for repository in (LocalContinuationRepository, LocalContinuationLeaseRepository):
        for name, member in inspect.getmembers(repository, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (repository.__name__, name)
            assert positions == tuple(sorted(positions)), (repository.__name__, name)
