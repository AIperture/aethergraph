from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    ExternalSessionBindingRequest,
    IngressClaimRequest,
    IngressClaimStatus,
    SessionKind,
    SessionRecord,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalIngressIdempotencyRepository,
    LocalIntegrationSessionRepository,
    LocalSessionRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 16, 1, tzinfo=UTC)
HOST_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
)
EXTERNAL_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    scope_key='{"conversation_id":"C1","thread_id":"T1"}',
)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _claim(
    idempotency_key: str = "key-1",
    *,
    external_event_id: str = "event-1",
    digest: str = "a" * 64,
    claimed_at: datetime = NOW,
    scope: StorageScope = HOST_SCOPE,
) -> IngressClaimRequest:
    return IngressClaimRequest(
        deployment_id="deployment-1",
        integration_id="slack-main",
        idempotency_key=idempotency_key,
        external_event_id=external_event_id,
        envelope_digest=digest,
        digest_algorithm="sha256",
        scope=scope,
        claimed_at=claimed_at,
    )


def _binding(
    *,
    binding_id: str = "binding-1",
    route_id: str = "route-1",
    build_id: str = "build-1",
    ag_session_id: str = "session-1",
    scope: StorageScope = EXTERNAL_SCOPE,
    now: datetime = NOW,
) -> ExternalSessionBindingRequest:
    return ExternalSessionBindingRequest(
        binding_id=binding_id,
        route_id=route_id,
        build_id=build_id,
        ag_session_id=ag_session_id,
        scope=scope,
        now=now,
    )


def _session(session_id: str = "session-1") -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        kind=SessionKind.CHAT,
        scope=StorageScope(
            tenant_id="tenant-1",
            project_id="project-1",
            org_id="org-1",
            user_id="user-1",
            session_id=session_id,
        ),
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        source="slack",
        external_reference="integration:route-1",
    )


@pytest.mark.asyncio
async def test_ingress_claim_replay_scope_and_single_assignment_receipt(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalIngressIdempotencyRepository(database=database)
    request = _claim()

    acquired = await repository.claim(request)
    assert acquired.acquired
    assert acquired.record.status is IngressClaimStatus.PENDING
    replay = await repository.claim(replace(request, claimed_at=NOW + timedelta(minutes=1)))
    assert not replay.acquired
    assert replay.record == acquired.record
    assert (
        await repository.get(
            HOST_SCOPE,
            request.deployment_id,
            request.integration_id,
            request.idempotency_key,
        )
        == acquired.record
    )
    assert (
        await repository.get(
            StorageScope(project_id="other"),
            request.deployment_id,
            request.integration_id,
            request.idempotency_key,
        )
        is None
    )

    completed = replace(
        acquired.record,
        status=IngressClaimStatus.COMPLETED,
        revision=2,
        receipt={"accepted": True, "outputs": ["one"]},
        completed_at=NOW + timedelta(seconds=1),
    )
    stored = await repository.complete(completed, 1)
    assert stored == completed
    replay = await repository.claim(request)
    assert not replay.acquired and replay.record == completed
    with pytest.raises(StorageConflictError, match="already assigned"):
        await repository.complete(replace(completed, revision=3), 2)
    await database.close()


@pytest.mark.asyncio
async def test_ingress_key_and_external_event_conflicts_are_fail_closed(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalIngressIdempotencyRepository(database=database)
    request = _claim()
    await repository.claim(request)

    with pytest.raises(StorageIntegrityError, match="idempotency identity"):
        await repository.claim(replace(request, envelope_digest="b" * 64))
    with pytest.raises(StorageIntegrityError, match="external event"):
        await repository.claim(_claim("key-2", external_event_id=request.external_event_id))
    assert (
        await repository.get(
            HOST_SCOPE,
            request.deployment_id,
            request.integration_id,
            "key-2",
        )
        is None
    )

    completed = replace(
        (await repository.claim(request)).record,
        status=IngressClaimStatus.COMPLETED,
        revision=2,
        receipt={"accepted": True},
        completed_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(StorageIntegrityError, match="immutable"):
        await repository.complete(replace(completed, external_event_id="other"), 1)
    with pytest.raises(StorageConflictError, match="stale"):
        await repository.complete(replace(completed, revision=1), 0)
    await database.close()


@pytest.mark.asyncio
async def test_concurrent_ingress_claim_has_exactly_one_owner(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalIngressIdempotencyRepository(database=database)
    results = await asyncio.gather(*(repository.claim(_claim()) for _ in range(20)))
    assert sum(result.acquired for result in results) == 1
    assert all(result.record == results[0].record for result in results)
    await database.close()


@pytest.mark.asyncio
async def test_integration_session_create_replay_and_monotonic_last_seen(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalIntegrationSessionRepository(database=database)
    request = _binding()

    created = await repository.provision(request, _session())
    assert created.binding_created and created.session_created
    assert created.binding.revision == 1
    replay = await repository.provision(request, _session())
    assert not replay.binding_created and not replay.session_created
    assert replay.binding == created.binding
    assert await repository.get_binding(EXTERNAL_SCOPE, request.route_id) == created.binding
    assert (
        await repository.get_binding(replace(EXTERNAL_SCOPE, project_id="other"), request.route_id)
        is None
    )
    with pytest.raises(ValueError, match="scope_key"):
        await repository.get_binding(HOST_SCOPE, request.route_id)

    later = await repository.provision(replace(request, now=NOW + timedelta(seconds=1)), _session())
    assert not later.binding_created
    assert later.binding.revision == 2
    assert later.binding.created_at == NOW
    assert later.binding.last_seen_at == NOW + timedelta(seconds=1)
    with pytest.raises(StorageIntegrityError, match="moved backward"):
        await repository.provision(request, _session())
    await database.close()


@pytest.mark.asyncio
async def test_integration_session_identity_is_pinned_and_creation_is_atomic(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalIntegrationSessionRepository(database=database)
    request = _binding()
    results = await asyncio.gather(*(repository.provision(request, _session()) for _ in range(20)))
    assert sum(result.binding_created for result in results) == 1
    assert sum(result.session_created for result in results) == 1
    assert all(result.binding.binding_id == results[0].binding.binding_id for result in results)

    for conflicting in (
        replace(request, binding_id="binding-other"),
        replace(request, build_id="build-other"),
        replace(request, ag_session_id="session-other"),
    ):
        with pytest.raises(StorageIntegrityError, match="identity conflicts"):
            await repository.provision(conflicting, _session(conflicting.ag_session_id))
    with pytest.raises(StorageIntegrityError, match="binding identity"):
        conflicting = _binding(binding_id=request.binding_id, route_id="route-other")
        await repository.provision(
            conflicting,
            replace(_session(), external_reference="integration:route-other"),
        )
    await database.close()


@pytest.mark.asyncio
async def test_provision_repairs_orphan_binding_before_artifact_accounting(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    integrations = LocalIntegrationSessionRepository(database=database)
    sessions = LocalSessionRepository(database=database)
    request = _binding()
    initial = await integrations.provision(request, _session())
    assert initial.binding_created and initial.session_created
    await database.transaction(
        lambda connection: connection.execute(
            "DELETE FROM local_sessions WHERE session_id = ?",
            (request.ag_session_id,),
        )
    )

    repaired = await integrations.provision(
        replace(request, now=NOW + timedelta(seconds=1)),
        _session(),
    )
    assert repaired.session_created
    assert not repaired.binding_created
    counted = await sessions.record_artifact(
        _session().scope,
        request.ag_session_id,
        "occurrence-1",
        NOW + timedelta(seconds=2),
    )
    assert counted.artifact_count == 1
    await database.close()


@pytest.mark.asyncio
async def test_integration_repositories_read_only_and_typed_corruption(tmp_path: Path) -> None:
    writer_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    ingress_writer = LocalIngressIdempotencyRepository(database=writer_database)
    binding_writer = LocalIntegrationSessionRepository(database=writer_database)
    request = _claim()
    binding = _binding()
    await ingress_writer.claim(request)
    await binding_writer.provision(binding, _session())
    await writer_database.close()

    database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    ingress = LocalIngressIdempotencyRepository(database=database)
    bindings = LocalIntegrationSessionRepository(database=database)
    assert (
        await ingress.get(
            HOST_SCOPE,
            request.deployment_id,
            request.integration_id,
            request.idempotency_key,
        )
        is not None
    )
    assert await bindings.get_binding(EXTERNAL_SCOPE, binding.route_id) is not None
    with pytest.raises(StorageReadOnlyError):
        await ingress.claim(_claim("new", external_event_id="new"))
    with pytest.raises(StorageReadOnlyError):
        new_binding = _binding(route_id="new", binding_id="new", ag_session_id="new")
        await bindings.provision(new_binding, _session("new"))
    await database.close()

    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    ingress = LocalIngressIdempotencyRepository(database=database)
    bindings = LocalIntegrationSessionRepository(database=database)
    await database.transaction(
        lambda connection: connection.execute("UPDATE local_ingress_claims SET receipt_json = '[]'")
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await ingress.get(
            HOST_SCOPE,
            request.deployment_id,
            request.integration_id,
            request.idempotency_key,
        )
    await database.transaction(
        lambda connection: connection.execute(
            "UPDATE local_external_session_bindings SET metadata_json = '[]'"
        )
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await bindings.get_binding(EXTERNAL_SCOPE, binding.route_id)
    await database.close()


@pytest.mark.asyncio
async def test_integration_schema_has_clean_identity_and_indexed_lookup(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalIngressIdempotencyRepository(database=database)
    LocalIntegrationSessionRepository(database=database)
    tables = (
        "local_ingress_claims",
        "local_external_session_bindings",
    )
    forbidden = {"app_id", "application_id", "client_id", "path", "route_json"}
    for table in tables:
        columns = await database.fetch_all(f"PRAGMA table_info({table})")
        assert forbidden.isdisjoint({str(column["name"]) for column in columns})

    ingress_plan = await database.fetch_all(
        "EXPLAIN QUERY PLAN SELECT * FROM local_ingress_claims "
        "WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?",
        ("deployment-1", "slack-main", "key-1"),
    )
    binding_plan = await database.fetch_all(
        "EXPLAIN QUERY PLAN SELECT * FROM local_external_session_bindings "
        "WHERE route_id = ? AND scope_key = ?",
        ("route-1", EXTERNAL_SCOPE.scope_key),
    )
    ingress_detail = " ".join(str(row[3]) for row in ingress_plan)
    binding_detail = " ".join(str(row[3]) for row in binding_plan)
    assert "INDEX" in ingress_detail and "SCAN" not in ingress_detail
    assert "ix_local_external_bindings_route_scope" in binding_detail
    await database.close()


def test_local_integration_public_docstrings_follow_repository_format() -> None:
    for repository in (
        LocalIngressIdempotencyRepository,
        LocalIntegrationSessionRepository,
    ):
        for name, method in inspect.getmembers(repository, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(method) or ""
            assert docstring.index("Examples:") < docstring.index("Args:")
            assert docstring.index("Args:") < docstring.index("Returns:")
            assert docstring.index("Returns:") < docstring.index("Notes:")
            assert docstring.count("```python") >= 2
