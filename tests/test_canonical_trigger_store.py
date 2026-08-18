from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from inspect import getdoc
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.api.v1.triggers import TriggerCreateRequest, TriggerMeta
from aethergraph.services.scope.scope import Scope
from aethergraph.services.triggers import CanonicalTriggerStore, bind_canonical_trigger_store
from aethergraph.services.triggers.trigger_service import TriggerServiceImpl
from aethergraph.services.triggers.types import TriggerRecord
from aethergraph.storage.contracts import (
    StorageConflictError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalSQLiteDatabase,
    LocalTriggerRepository,
)

NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)
OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self, value: datetime = NOW + timedelta(seconds=1)) -> None:
        self.value = value

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(microseconds=1)
        return current


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _store(
    root: Path,
) -> tuple[CanonicalTriggerStore, LocalTriggerRepository, LocalSQLiteDatabase, _Clock]:
    database = _database(root)
    repository = LocalTriggerRepository(database=database)
    clock = _Clock()
    return (
        CanonicalTriggerStore(repository=repository, owner_scope=OWNER, clock=clock.now),
        repository,
        database,
        clock,
    )


def _trigger(
    trigger_id: str,
    *,
    kind: str = "interval",
    client_id: str = "client-1",
    app_id: str | None = "compat-app",
    next_fire_at: datetime | None = NOW,
) -> TriggerRecord:
    record = TriggerRecord(
        trigger_id=trigger_id,
        trigger_name="Daily report",
        org_id="org-1",
        user_id="user-1",
        client_id=client_id,
        mode="demo",
        app_id=app_id,
        agent_id="agent-1",
        session_id="session-1",
        memory_level="session",
        graph_id="graph-1",
        default_inputs={"messages": ["hello"]},
        origin="schedule",
        kind=kind,  # type: ignore[arg-type]
        interval_seconds=60 if kind == "interval" else None,
        event_key="invoice.paid" if kind == "event" else None,
        active=True,
        created_at=NOW,
        next_fire_at=None if kind == "event" else next_fire_at,
        max_overlap_runs=0,
        meta={"display": {"color": "blue"}},
    )
    return record


@pytest.mark.asyncio
async def test_canonical_trigger_projects_scope_and_explicit_app_compatibility_metadata(
    tmp_path: Path,
) -> None:
    store, repository, database, _clock = _store(tmp_path)
    trigger = _trigger("trigger-1")

    await store.create(trigger)
    await store.create(trigger)
    canonical = await repository.get(OWNER, trigger.trigger_id)
    restored = await store.get(trigger.trigger_id)
    columns = {
        str(row["name"]) for row in await database.fetch_all("PRAGMA table_info(local_triggers)")
    }

    assert canonical is not None and restored is not None
    assert canonical.scope == StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
        session_id="session-1",
        graph_id="graph-1",
        agent_id="agent-1",
    )
    assert canonical.max_overlap_runs == 0
    assert "app_id" not in canonical.scope.as_filter()
    assert "client_id" not in canonical.scope.as_filter()
    assert canonical.metadata["compatibility_metadata"]["app_id"] == {
        "value": "compat-app",
        "deprecated": True,
        "scheduled_removal": "future breaking release",
    }
    assert canonical.metadata["service_context"]["client_id"] == "client-1"
    assert not {"app_id", "application_id", "client_id"} & columns
    assert restored.app_id == "compat-app"
    assert restored.client_id == "client-1"
    assert restored.default_inputs == {"messages": ["hello"]}
    assert restored.meta == {"display": {"color": "blue"}}
    await database.close()


@pytest.mark.asyncio
async def test_canonical_trigger_updates_use_exact_attached_revision(tmp_path: Path) -> None:
    store, repository, database, _clock = _store(tmp_path)
    await store.create(_trigger("trigger-cas"))
    first = await store.get("trigger-cas")
    stale = await store.get("trigger-cas")
    assert first is not None and stale is not None

    first.active = False
    first.next_fire_at = None
    await store.update(first)
    stale.trigger_name = "stale rename"
    with pytest.raises(StorageConflictError, match="stale"):
        await store.update(stale)
    with pytest.raises(StorageConflictError, match="repository-authored"):
        await store.update(_trigger("trigger-cas"))
    persisted = await repository.get(OWNER, "trigger-cas")
    assert persisted is not None
    assert persisted.revision == 2
    assert persisted.active is False
    await database.close()


@pytest.mark.asyncio
async def test_trigger_service_consumes_canonical_store_without_app_or_client_partition(
    tmp_path: Path,
) -> None:
    store, _repository, database, _clock = _store(tmp_path)
    service = TriggerServiceImpl(store=store)
    scope = Scope(
        org_id="org-1",
        user_id="user-1",
        client_id="client-1",
        mode="demo",
        app_id="compat-app",
        agent_id="agent-1",
        session_id="session-1",
    )

    trigger = await service.create_from_scope(
        scope=scope,
        graph_id="graph-1",
        default_inputs={"message": "hello"},
        kind="interval",
        interval_seconds=60,
        max_overlap_runs=0,
    )
    owned = await service.list_for_owner(org_id="org-1", user_id="user-1")
    assert [record.trigger_id for record in owned] == [trigger.trigger_id]
    assert owned[0].app_id == "compat-app"
    assert await service.cancel(
        trigger.trigger_id,
        org_id="org-1",
        user_id="user-1",
        client_id="client-1",
    )
    canceled = await store.get(trigger.trigger_id)
    assert canceled is not None and canceled.active is False
    await database.close()


@pytest.mark.asyncio
async def test_canonical_event_queries_filter_client_only_after_bounded_hydration(
    tmp_path: Path,
) -> None:
    store, _repository, database, _clock = _store(tmp_path)
    await store.create(_trigger("event-a", kind="event", client_id="client-a"))
    await store.create(_trigger("event-b", kind="event", client_id="client-b"))

    by_client = await store.list_by_event_key("invoice.paid", client_id="client-a")
    by_user_alias = await store.list_all(user_id="client-b", kind="event", active=True)

    assert [record.trigger_id for record in by_client] == ["event-a"]
    assert [record.trigger_id for record in by_user_alias] == ["event-b"]
    with pytest.raises(ValueError, match="explicit tenant scope"):
        await store.list_by_event_key("invoice.paid")
    await database.close()


@pytest.mark.asyncio
async def test_canonical_claim_transitions_and_receipts_survive_restart(tmp_path: Path) -> None:
    store, _repository, database, _clock = _store(tmp_path)
    for trigger_id in ("trigger-complete", "trigger-fail", "trigger-skip"):
        await store.create(_trigger(trigger_id, app_id=None))

    claims = await store.claim_due(
        NOW,
        worker_id="worker-1",
        lease_until=NOW + timedelta(seconds=30),
        limit=10,
    )
    by_trigger = {claim.trigger.trigger_id: claim for claim in claims}
    completed = by_trigger["trigger-complete"]
    failed = by_trigger["trigger-fail"]
    skipped = by_trigger["trigger-skip"]
    assert await store.complete_claim(
        completed.fire_id,
        worker_id="worker-1",
        run_id="run-1",
        completed_at=NOW + timedelta(seconds=1),
    )
    assert await store.fail_claim(
        failed.fire_id,
        worker_id="worker-1",
        error="runner offline",
        retry_at=NOW + timedelta(seconds=5),
    )
    assert await store.skip_claim(
        skipped.fire_id,
        worker_id="worker-1",
        reason="overlap",
        completed_at=NOW + timedelta(seconds=1),
    )
    assert not await store.complete_claim(
        completed.fire_id,
        worker_id="stale-worker",
        run_id="run-2",
        completed_at=NOW + timedelta(seconds=2),
    )
    await database.close()

    reopened = _database(tmp_path)
    restarted = CanonicalTriggerStore(
        repository=LocalTriggerRepository(database=reopened),
        owner_scope=OWNER,
        clock=_Clock(NOW + timedelta(seconds=5)).now,
    )
    complete_receipt = await restarted.get_claim(completed.fire_id)
    skip_receipt = await restarted.get_claim(skipped.fire_id)
    retried = await restarted.claim_due(
        NOW + timedelta(seconds=5),
        worker_id="worker-2",
        lease_until=NOW + timedelta(seconds=35),
        limit=1,
    )

    assert complete_receipt is not None
    assert complete_receipt["status"] == "delivered"
    assert complete_receipt["run_id"] == "run-1"
    assert skip_receipt is not None and skip_receipt["status"] == "skipped_overlap"
    assert len(retried) == 1
    assert retried[0].fire_id == failed.fire_id
    assert retried[0].attempts == 2
    assert retried[0].reclaimed is False
    await reopened.close()


def test_canonical_trigger_factory_maps_only_exact_bundle_field_without_io() -> None:
    repository = object()
    bundle = SimpleNamespace(triggers=repository)

    store = bind_canonical_trigger_store(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=OWNER,
        clock=lambda: NOW,
    )

    assert store._repository is repository
    assert store._owner_scope == OWNER


def test_trigger_app_id_is_explicitly_deprecated_compatibility_metadata() -> None:
    service_field = next(item for item in fields(TriggerRecord) if item.name == "app_id")

    assert service_field.metadata["deprecated"] is True
    assert "future breaking release" in service_field.metadata["description"]
    assert TriggerCreateRequest.model_fields["app_id"].deprecated is True
    assert TriggerMeta.model_fields["app_id"].deprecated is True


def test_canonical_trigger_public_docstrings_follow_strict_contract() -> None:
    methods = (
        CanonicalTriggerStore.__init__,
        CanonicalTriggerStore.create,
        CanonicalTriggerStore.update,
        CanonicalTriggerStore.get,
        CanonicalTriggerStore.delete,
        CanonicalTriggerStore.claim_due,
        CanonicalTriggerStore.complete_claim,
        CanonicalTriggerStore.fail_claim,
        CanonicalTriggerStore.skip_claim,
        CanonicalTriggerStore.get_claim,
        CanonicalTriggerStore.list_all,
        CanonicalTriggerStore.list_by_event_key,
        bind_canonical_trigger_store,
    )
    for method in methods:
        docstring = getdoc(method)
        assert docstring is not None
        assert docstring.count("```python") == 2
        positions = [
            docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
        ]
        assert positions == sorted(positions)
