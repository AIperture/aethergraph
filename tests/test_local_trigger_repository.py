from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
    TriggerClaimRequest,
    TriggerClaimStatus,
    TriggerKind,
    TriggerQuery,
    TriggerRecord,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalSQLiteDatabase,
    LocalTriggerRepository,
)

NOW = datetime(2026, 8, 15, 22, tzinfo=UTC)
SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
    graph_id="graph-1",
)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _trigger(
    trigger_id: str,
    *,
    scope: StorageScope = SCOPE,
    kind: TriggerKind = TriggerKind.INTERVAL,
    created_at: datetime = NOW,
    updated_at: datetime = NOW,
    next_fire_at: datetime | None = NOW,
    catch_up_missed: bool = False,
) -> TriggerRecord:
    schedule: dict[str, object] = {}
    if kind is TriggerKind.INTERVAL:
        schedule["interval_seconds"] = 10
    elif kind is TriggerKind.CRON:
        schedule.update(cron_expression="0 9 * * *", timezone="America/Los_Angeles")
    elif kind is TriggerKind.ONE_SHOT:
        schedule["run_at"] = next_fire_at
    else:
        schedule["event_key"] = "invoice.paid"
    return TriggerRecord(
        trigger_id=trigger_id,
        graph_id=scope.graph_id or "",
        scope=scope,
        kind=kind,
        revision=1,
        created_at=created_at,
        updated_at=updated_at,
        next_fire_at=next_fire_at,
        catch_up_missed=catch_up_missed,
        default_inputs={"message": "hello"},
        metadata={"source": "test"},
        **schedule,
    )


def _request(
    *,
    now: datetime = NOW,
    worker_id: str = "worker-1",
    limit: int = 10,
    scope: StorageScope | None = None,
    skip_missed_before: datetime | None = None,
) -> TriggerClaimRequest:
    return TriggerClaimRequest(
        now=now,
        worker_id=worker_id,
        lease_until=now + timedelta(seconds=30),
        limit=limit,
        scope=scope,
        skip_missed_before=skip_missed_before,
    )


@pytest.mark.asyncio
async def test_trigger_create_get_cas_and_bound_cursor_query(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    records = tuple(
        _trigger(
            f"trigger-{index}",
            created_at=NOW + timedelta(seconds=index),
            updated_at=NOW + timedelta(seconds=index),
            next_fire_at=NOW + timedelta(minutes=1, seconds=index),
        )
        for index in range(3)
    )
    for record in records:
        assert await repository.create(record) == record
    assert await repository.create(records[0]) == records[0]
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.create(replace(records[0], name="different"))
    assert await repository.get(SCOPE, records[0].trigger_id) == records[0]
    assert await repository.get(StorageScope(project_id="other"), records[0].trigger_id) is None
    assert await repository.get(StorageScope(), records[0].trigger_id) is None

    query = TriggerQuery(scope=SCOPE, kinds=(TriggerKind.INTERVAL,), page=PageRequest(limit=2))
    first = await repository.query(query)
    second = await repository.query(
        replace(query, page=PageRequest(limit=2, cursor=first.next_cursor))
    )
    assert (*first.items, *second.items) == tuple(reversed(records))
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query(
            replace(
                query,
                active=True,
                page=PageRequest(limit=2, cursor=first.next_cursor),
            )
        )

    paused = replace(
        records[0],
        revision=2,
        updated_at=NOW + timedelta(minutes=2),
        active=False,
        next_fire_at=None,
    )
    assert await repository.compare_and_set(paused, 1) == paused
    with pytest.raises(StorageConflictError):
        await repository.compare_and_set(paused, 1)
    with pytest.raises(StorageIntegrityError, match="immutable"):
        await repository.compare_and_set(replace(paused, revision=3, origin="other"), 2)
    await database.close()


@pytest.mark.asyncio
async def test_multi_worker_due_claim_is_atomic_and_advances_schedule(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    await repository.create(_trigger("trigger-1"))

    batches = await asyncio.gather(
        *(repository.claim_due(_request(worker_id=f"worker-{index}")) for index in range(20))
    )
    claims = tuple(claim for batch in batches for claim in batch)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim.attempts == claim.claim.revision == 1
    assert claim.claim.scheduled_for == NOW
    stored = await repository.get(SCOPE, "trigger-1")
    assert stored is not None
    assert stored.revision == 2
    assert stored.next_fire_at == NOW + timedelta(seconds=10)
    await database.close()


@pytest.mark.asyncio
async def test_expired_trigger_lease_reports_exact_reclaim_evidence(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    await repository.create(_trigger("trigger-stale"))
    initial = (await repository.claim_due(_request(limit=1)))[0]

    recovered = await repository.claim_due(
        _request(
            now=initial.claim.lease_until + timedelta(seconds=1),
            worker_id="recovery-worker",
            limit=1,
        )
    )

    assert len(recovered) == 1
    assert recovered[0].claim.fire_id == initial.claim.fire_id
    assert recovered[0].reclaimed is True
    await database.close()


@pytest.mark.asyncio
async def test_retry_reclaim_precedes_new_due_and_delivery_updates_trigger(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    await repository.create(_trigger("trigger-retry"))
    initial = (await repository.claim_due(_request(limit=1)))[0]
    retry_at = NOW + timedelta(seconds=5)
    retry = replace(
        initial.claim,
        revision=2,
        status=TriggerClaimStatus.RETRY,
        updated_at=NOW + timedelta(seconds=1),
        worker_id=None,
        lease_until=None,
        retry_at=retry_at,
        last_error="runner unavailable",
    )
    assert await repository.compare_and_set_claim(retry, 1) == retry
    await repository.create(_trigger("trigger-new", next_fire_at=retry_at))

    batch = await repository.claim_due(_request(now=retry_at, worker_id="worker-2", limit=1))
    assert len(batch) == 1
    reclaimed = batch[0]
    assert reclaimed.reclaimed is False
    assert reclaimed.claim.fire_id == initial.claim.fire_id
    assert reclaimed.claim.attempts == 2
    assert reclaimed.claim.revision == 3
    assert reclaimed.claim.worker_id == "worker-2"

    delivered_at = retry_at + timedelta(seconds=1)
    delivered = replace(
        reclaimed.claim,
        revision=4,
        status=TriggerClaimStatus.DELIVERED,
        updated_at=delivered_at,
        worker_id=None,
        lease_until=None,
        run_id="run-1",
        finished_at=delivered_at,
    )
    assert await repository.compare_and_set_claim(delivered, 3) == delivered
    trigger = await repository.get(SCOPE, "trigger-retry")
    assert trigger is not None
    assert trigger.revision == 3
    assert trigger.last_fired_at == delivered_at
    with pytest.raises(StorageConflictError, match="active"):
        await repository.compare_and_set_claim(replace(delivered, revision=5), 4)
    await database.close()


@pytest.mark.asyncio
async def test_missed_and_catch_up_policies_commit_receipts_and_revisions(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    skipped = _trigger("trigger-skip")
    catch_up = _trigger("trigger-catch", catch_up_missed=True)
    await repository.create(skipped)
    await repository.create(catch_up)
    now = NOW + timedelta(seconds=35)

    claims = await repository.claim_due(_request(now=now, skip_missed_before=now, limit=10))
    assert [item.trigger.trigger_id for item in claims] == ["trigger-catch"]
    stored_skip = await repository.get(SCOPE, skipped.trigger_id)
    stored_catch = await repository.get(SCOPE, catch_up.trigger_id)
    assert stored_skip is not None and stored_catch is not None
    assert stored_skip.revision == 2
    assert stored_skip.next_fire_at == NOW + timedelta(seconds=40)
    assert stored_catch.revision == 2
    assert stored_catch.next_fire_at == NOW + timedelta(seconds=10)

    rows = await database.fetch_all(
        "SELECT fire_id FROM local_trigger_claims WHERE trigger_id = ?",
        (skipped.trigger_id,),
    )
    receipt = await repository.get_claim(SCOPE, str(rows[0]["fire_id"]))
    assert receipt is not None
    assert receipt.status is TriggerClaimStatus.SKIPPED
    assert receipt.attempts == 0
    assert receipt.skip_reason == "missed_before_startup"
    await database.close()


@pytest.mark.asyncio
async def test_overdue_zero_overlap_one_shot_survives_restart_and_skips_exactly_once(
    tmp_path: Path,
) -> None:
    overdue = NOW - timedelta(minutes=5)
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    record = replace(
        _trigger(
            "trigger-overdue",
            kind=TriggerKind.ONE_SHOT,
            created_at=NOW,
            updated_at=NOW,
            next_fire_at=overdue,
        ),
        max_overlap_runs=0,
    )
    await repository.create(record)
    await database.close()

    reopened = _database(tmp_path, StorageOpenMode.READ_WRITE)
    restarted_repository = LocalTriggerRepository(database=reopened)
    restored = await restarted_repository.get(SCOPE, record.trigger_id)
    assert restored == record
    claims = await restarted_repository.claim_due(
        _request(now=NOW, skip_missed_before=NOW, limit=10)
    )
    assert claims == ()
    skipped = await restarted_repository.get(SCOPE, record.trigger_id)
    assert skipped is not None
    assert skipped.max_overlap_runs == 0
    assert skipped.active is False
    assert skipped.next_fire_at is None
    receipts = await reopened.fetch_all(
        "SELECT fire_id FROM local_trigger_claims WHERE trigger_id = ?",
        (record.trigger_id,),
    )
    assert len(receipts) == 1
    receipt = await restarted_repository.get_claim(SCOPE, str(receipts[0]["fire_id"]))
    assert receipt is not None
    assert receipt.status is TriggerClaimStatus.SKIPPED
    assert receipt.skip_reason == "missed_before_startup"
    assert (
        await restarted_repository.claim_due(
            _request(now=NOW + timedelta(minutes=1), skip_missed_before=NOW, limit=10)
        )
        == ()
    )
    await reopened.close()


@pytest.mark.asyncio
async def test_scoped_claim_never_falls_back_to_another_owner(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    other_scope = replace(SCOPE, project_id="project-2", graph_id="graph-2")
    await repository.create(_trigger("owned", scope=SCOPE))
    await repository.create(_trigger("other", scope=other_scope))

    claims = await repository.claim_due(_request(scope=SCOPE))
    assert [item.trigger.trigger_id for item in claims] == ["owned"]
    assert await repository.get(SCOPE, "other") is None
    other = await repository.get(other_scope, "other")
    assert other is not None and other.next_fire_at == NOW
    await database.close()


@pytest.mark.asyncio
async def test_delete_cleans_nonterminal_claim_and_preserves_terminal_receipt(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    await repository.create(_trigger("terminal"))
    terminal_claim = (await repository.claim_due(_request(limit=1)))[0]
    finished_at = NOW + timedelta(seconds=1)
    skipped = replace(
        terminal_claim.claim,
        revision=2,
        status=TriggerClaimStatus.SKIPPED,
        updated_at=finished_at,
        worker_id=None,
        lease_until=None,
        skip_reason="overlap_limit",
        finished_at=finished_at,
    )
    await repository.compare_and_set_claim(skipped, 1)
    terminal = await repository.get(SCOPE, "terminal")
    assert terminal is not None
    assert await repository.delete(SCOPE, "terminal", terminal.revision)
    assert await repository.get_claim(SCOPE, skipped.fire_id) == skipped
    with pytest.raises(StorageIntegrityError, match="retained receipts"):
        await repository.create(_trigger("terminal"))

    await repository.create(_trigger("active"))
    active_claim = (await repository.claim_due(_request(limit=1)))[0]
    active = await repository.get(SCOPE, "active")
    assert active is not None
    assert await repository.delete(SCOPE, "active", active.revision)
    assert await repository.get_claim(SCOPE, active_claim.claim.fire_id) is None
    await database.close()


@pytest.mark.asyncio
async def test_event_query_schedule_validation_and_query_indexes(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    event = _trigger("event-1", kind=TriggerKind.EVENT, next_fire_at=None)
    await repository.create(event)
    page = await repository.query(
        TriggerQuery(
            scope=SCOPE,
            kinds=(TriggerKind.EVENT,),
            active=True,
            event_key="invoice.paid",
        )
    )
    assert page.items == (event,)
    invalid_cron = replace(
        _trigger("cron-1", kind=TriggerKind.CRON, next_fire_at=NOW + timedelta(minutes=1)),
        cron_expression="invalid cron",
    )
    with pytest.raises(StorageConfigurationError, match="cron"):
        await repository.create(invalid_cron)
    invalid_timezone = replace(event, trigger_id="event-2", timezone="Mars/Olympus")
    with pytest.raises(StorageConfigurationError, match="timezone"):
        await repository.create(invalid_timezone)

    due_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_triggers
            WHERE active = 1 AND kind != 'event' AND next_fire_at IS NOT NULL
              AND next_fire_at <= ?
            ORDER BY next_fire_at ASC, trigger_id ASC LIMIT ?
            """,
            (NOW.isoformat(), 20),
        )
    )
    event_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_triggers
            WHERE event_key = ? AND active = 1 AND kind = 'event'
            ORDER BY updated_at DESC, trigger_id DESC LIMIT ?
            """,
            ("invoice.paid", 20),
        )
    )
    claim_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_trigger_claims
            WHERE status = 'retry' AND retry_at <= ?
            ORDER BY scheduled_for ASC, fire_id ASC LIMIT ?
            """,
            (NOW.isoformat(), 20),
        )
    )
    assert "ix_local_triggers_due" in due_plan
    assert "ix_local_triggers_event_lookup" in event_plan
    assert "ix_local_trigger_claims_eligible" in claim_plan
    assert "SCAN local_" not in f"{due_plan} {event_plan} {claim_plan}"
    await database.close()


@pytest.mark.asyncio
async def test_trigger_schema_is_canonical_and_read_only_rejects_mutation(
    tmp_path: Path,
) -> None:
    writable = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=writable)
    record = _trigger("trigger-1")
    await repository.create(record)
    for table in ("local_triggers", "local_trigger_claims"):
        columns = {
            str(row["name"]) for row in await writable.fetch_all(f"PRAGMA table_info({table})")
        }
        assert not {"app_id", "application_id", "client_id", "path"} & columns
    await writable.close()

    readonly = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly_repository = LocalTriggerRepository(database=readonly)
    assert await readonly_repository.get(SCOPE, record.trigger_id) == record
    with pytest.raises(StorageReadOnlyError):
        await readonly_repository.create(_trigger("new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly_repository.claim_due(_request())
    with pytest.raises(StorageReadOnlyError):
        await readonly_repository.delete(SCOPE, record.trigger_id, record.revision)
    await readonly.close()


@pytest.mark.asyncio
async def test_persisted_trigger_corruption_is_typed(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalTriggerRepository(database=database)
    await repository.create(_trigger("trigger-1"))
    await database.execute(
        "UPDATE local_triggers SET metadata_json = ? WHERE trigger_id = ?",
        ("[]", "trigger-1"),
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await repository.get(SCOPE, "trigger-1")
    await database.close()


def test_local_trigger_docstrings_follow_required_complete_format() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for name, member in inspect.getmembers(LocalTriggerRepository, inspect.isfunction):
        if name.startswith("_"):
            continue
        docstring = inspect.getdoc(member) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), name
        assert positions == tuple(sorted(positions)), name
        assert docstring.count("```python") >= 2, name
