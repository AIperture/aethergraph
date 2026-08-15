from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from inspect import getdoc
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.integration import (
    ExternalIdentity,
    ExternalSessionBinding,
    IngressEnvelope,
    IngressReceipt,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    MessageCompletedPayload,
    OriginAddress,
    SemanticEvent,
    SemanticEventKind,
)
from aethergraph.services.channel.resources import InputResource
from aethergraph.services.integration import (
    CanonicalExternalSessionBindingStore,
    CanonicalInboundEventStore,
    CanonicalIngressIdempotencyStore,
    CanonicalIntegrationPersistence,
    CanonicalSemanticEventStore,
    IngressIdempotencyError,
    SemanticEventStoreError,
    SessionBindingError,
    bind_canonical_integration_persistence,
)
from aethergraph.storage.contracts import (
    IngressClaimStatus,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalExternalSessionBindingRepository,
    LocalInboundEventRepository,
    LocalIngressIdempotencyRepository,
    LocalSemanticEventRepository,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 2, tzinfo=UTC)
_OWNER = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    org_id="org-1",
)


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _event_database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.EVENTS,
        mode=StorageOpenMode.READ_WRITE,
    )


def _route() -> IntegrationRoute:
    return IntegrationRoute(
        route_id="route-slack",
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        entry_agent_id="agent.support",
        enabled=True,
        match_policy=IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope="conversation_thread"),
        required_capabilities=IntegrationCapabilities(
            event_kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
            streaming=False,
            interactions=True,
            attachments=True,
            cancellation=True,
        ),
    )


def _identity(*, user_id: str = "user-1", thread_id: str | None = "thread-1") -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id="external-tenant",
        conversation_id="conversation-1",
        thread_id=thread_id,
        user_id=user_id,
    )


def _envelope(
    *,
    idempotency_key: str = "event-1",
    external_event_id: str = "event-1",
    text: str = "hello",
    received_at: datetime = _NOW,
) -> IngressEnvelope:
    return IngressEnvelope(
        integration_id="slack-main",
        route_hint="route-slack",
        external_identity=_identity(),
        external_event_id=external_event_id,
        idempotency_key=idempotency_key,
        received_at=received_at,
        text=text,
        origin_address=OriginAddress(
            channel_key="slack:conversation-1:thread-1",
            capability_profile_id="slack-v1",
        ),
    )


def _receipt() -> IngressReceipt:
    return IngressReceipt(
        accepted=True,
        duplicate=False,
        action="root_turn_started",
        deployment_id="deployment-1",
        route_id="route-slack",
        session_id="session-1",
        turn_id="turn-1",
        event_cursor=1,
    )


def _binding() -> ExternalSessionBinding:
    return ExternalSessionBinding(
        binding_id="binding-1",
        route_id="route-slack",
        external_identity=_identity(),
        ag_session_id="session-1",
        build_id="build-1",
        created_at=_NOW,
        last_seen_at=_NOW,
    )


def _semantic_event(
    *,
    event_id: str = "semantic-1",
    sequence: int = 0,
    timestamp: datetime = _NOW,
) -> SemanticEvent:
    return SemanticEvent(
        event_id=event_id,
        deployment_id="deployment-1",
        session_id="session-1",
        turn_id="turn-1",
        sequence=sequence,
        producer="agent.support",
        timestamp=timestamp,
        kind=SemanticEventKind.MESSAGE_COMPLETED,
        payload=MessageCompletedPayload(
            message_id=f"message-{sequence}",
            text=f"message {sequence}",
        ),
        extensions={"aethergraph.test": sequence},
    )


@pytest.mark.asyncio
async def test_canonical_ingress_projection_claims_completes_and_replays(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = LocalIngressIdempotencyRepository(database=database)
    store = CanonicalIngressIdempotencyStore(
        repository=repository,
        owner_scope=_OWNER,
        clock=lambda: _NOW + timedelta(seconds=1),
    )
    envelope = _envelope()

    claims = await asyncio.gather(
        *(store.claim(deployment_id="deployment-1", envelope=envelope) for _ in range(8))
    )
    assert sum(item.acquired for item in claims) == 1
    assert sum(item.pending for item in claims) == 7

    receipt = _receipt()
    await store.complete(
        deployment_id="deployment-1",
        envelope=envelope,
        receipt=receipt,
    )
    duplicate = await store.claim(
        deployment_id="deployment-1",
        envelope=_envelope(received_at=_NOW + timedelta(minutes=1)),
    )
    assert duplicate.receipt == receipt.model_copy(update={"duplicate": True})
    persisted = await repository.get(
        _OWNER,
        "deployment-1",
        envelope.integration_id,
        envelope.idempotency_key,
    )
    assert persisted is not None
    assert persisted.status is IngressClaimStatus.COMPLETED
    assert persisted.receipt["duplicate"] is False
    await database.close()


@pytest.mark.asyncio
async def test_canonical_ingress_projection_preserves_stable_host_failures(tmp_path: Path) -> None:
    database = _database(tmp_path)
    store = CanonicalIngressIdempotencyStore(
        repository=LocalIngressIdempotencyRepository(database=database),
        owner_scope=_OWNER,
        clock=lambda: _NOW + timedelta(seconds=1),
    )
    envelope = _envelope()
    await store.claim(deployment_id="deployment-1", envelope=envelope)

    with pytest.raises(IngressIdempotencyError) as conflict:
        await store.claim(
            deployment_id="deployment-1",
            envelope=_envelope(text="different"),
        )
    assert conflict.value.code == "integration.idempotency_conflict"

    with pytest.raises(IngressIdempotencyError) as missing:
        await store.complete(
            deployment_id="deployment-1",
            envelope=_envelope(idempotency_key="missing", external_event_id="missing"),
            receipt=_receipt(),
        )
    assert missing.value.code == "integration.idempotency_not_claimed"

    await store.complete(
        deployment_id="deployment-1",
        envelope=envelope,
        receipt=_receipt(),
    )
    with pytest.raises(IngressIdempotencyError) as repeated:
        await store.complete(
            deployment_id="deployment-1",
            envelope=envelope,
            receipt=_receipt(),
        )
    assert repeated.value.code == "integration.idempotency_already_completed"
    await database.close()


@pytest.mark.asyncio
async def test_canonical_binding_projection_uses_candidates_only_for_creation(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = LocalExternalSessionBindingRepository(database=database)
    store = CanonicalExternalSessionBindingStore(
        repository=repository,
        owner_scope=_OWNER,
    )
    route = _route()

    results = await asyncio.gather(
        *(
            store.get_or_create(
                route=route,
                external_identity=_identity(user_id=f"user-{index}"),
                build_id="build-1",
                binding_id=f"binding-{index}",
                ag_session_id=f"session-{index}",
                now=_NOW,
            )
            for index in range(8)
        )
    )
    assert sum(item.created for item in results) == 1
    assert len({item.binding.binding_id for item in results}) == 1
    assert len({item.binding.ag_session_id for item in results}) == 1

    resolved = await store.get_or_create(
        route=route,
        external_identity=_identity(user_id="another-user"),
        build_id="build-1",
        binding_id="unused-candidate",
        ag_session_id="unused-session",
        now=_NOW + timedelta(seconds=1),
    )
    assert not resolved.created
    assert resolved.binding.binding_id == results[0].binding.binding_id
    assert resolved.binding.external_identity.user_id == "another-user"

    with pytest.raises(SessionBindingError) as mismatch:
        await store.get_or_create(
            route=route,
            external_identity=_identity(),
            build_id="build-2",
            binding_id="replacement",
            ag_session_id="replacement",
            now=_NOW + timedelta(seconds=2),
        )
    assert mismatch.value.code == "integration.binding_build_mismatch"
    await database.close()


@pytest.mark.asyncio
async def test_canonical_binding_projection_requires_route_thread(tmp_path: Path) -> None:
    database = _database(tmp_path)
    store = CanonicalExternalSessionBindingStore(
        repository=LocalExternalSessionBindingRepository(database=database),
        owner_scope=_OWNER,
    )

    with pytest.raises(SessionBindingError) as missing:
        await store.get_or_create(
            route=_route(),
            external_identity=_identity(thread_id=None),
            build_id="build-1",
            binding_id="binding-1",
            ag_session_id="session-1",
            now=_NOW,
        )
    assert missing.value.code == "integration.binding_thread_required"
    await database.close()


@pytest.mark.asyncio
async def test_canonical_inbound_projection_normalizes_resources_and_delivery_cursor(
    tmp_path: Path,
) -> None:
    database = _event_database(tmp_path)
    repository = LocalInboundEventRepository(database=database)
    store = CanonicalInboundEventStore(repository=repository, owner_scope=_OWNER)
    resource = InputResource(
        kind="artifact",
        source="slack",
        status="materialized",
        name="report.txt",
        mime="text/plain",
        size=12,
        artifact_id="artifact-1",
        path="C:/provider-private/source.txt",
        uri="artifact://provider-private",
    )

    persisted = await store.append(
        deployment_id="deployment-1",
        route=_route(),
        binding=_binding(),
        envelope=_envelope(),
        resources=(resource,),
    )
    scope = StorageScope(**_OWNER.as_filter(), session_id="session-1")
    record = await repository.get(scope, persisted.event_id)

    assert record is not None
    assert persisted.cursor == record.delivery_cursor == 1
    assert record.resource_keys == ("artifact:artifact-1",)
    normalized_resource = record.payload["resources"][0]
    assert normalized_resource["artifact_id"] == "artifact-1"
    assert normalized_resource["media_type"] == "text/plain"
    assert "path" not in normalized_resource
    assert "uri" not in normalized_resource
    await database.close()


@pytest.mark.asyncio
async def test_canonical_semantic_projection_round_trips_and_resumes_by_delivery_cursor(
    tmp_path: Path,
) -> None:
    database = _event_database(tmp_path)
    repository = LocalSemanticEventRepository(database=database)
    store = CanonicalSemanticEventStore(repository=repository, owner_scope=_OWNER)
    events = tuple(
        _semantic_event(
            event_id=f"semantic-{index}",
            sequence=index,
            timestamp=_NOW + timedelta(seconds=index),
        )
        for index in range(3)
    )

    persisted = tuple([await store.append(event) for event in events])
    assert [item.cursor for item in persisted] == [1, 2, 3]
    assert [item.event for item in persisted] == list(events)
    resumed = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
        after_cursor=1,
        limit=2,
    )
    assert resumed == persisted[1:]
    bounded = CanonicalSemanticEventStore(
        repository=repository,
        owner_scope=_OWNER,
        max_history_events=2,
    )
    with pytest.raises(SemanticEventStoreError) as exceeded:
        await bounded.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
        )
    assert exceeded.value.code == "integration.semantic_event_history_limit"

    await database.close()
    reopened = _event_database(tmp_path)
    restored = CanonicalSemanticEventStore(
        repository=LocalSemanticEventRepository(database=reopened),
        owner_scope=_OWNER,
    )
    assert (
        await restored.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
            after_cursor=2,
            limit=10,
        )
        == persisted[2:]
    )
    await reopened.close()


def test_canonical_integration_factory_maps_exact_bundle_fields_without_io() -> None:
    ingress = object()
    bindings = object()
    bundle = SimpleNamespace(
        ingress_idempotency=ingress,
        external_session_bindings=bindings,
        inbound_events=object(),
        semantic_events=object(),
    )

    persistence = bind_canonical_integration_persistence(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=_OWNER,
        clock=lambda: _NOW,
    )

    assert isinstance(persistence.idempotency, CanonicalIngressIdempotencyStore)
    assert isinstance(persistence.bindings, CanonicalExternalSessionBindingStore)
    assert isinstance(persistence.inbound_events, CanonicalInboundEventStore)
    assert isinstance(persistence.semantic_events, CanonicalSemanticEventStore)
    assert persistence.idempotency._repository is ingress
    assert persistence.bindings._repository is bindings
    assert {field.name for field in fields(CanonicalIntegrationPersistence)} == {
        "idempotency",
        "bindings",
        "inbound_events",
        "semantic_events",
    }


@pytest.mark.parametrize(
    "scope",
    [
        StorageScope(),
        StorageScope(project_id="project-1", session_id="session-1"),
        StorageScope(project_id="project-1", run_id="run-1"),
        StorageScope(project_id="project-1", node_id="node-1"),
        StorageScope(project_id="project-1", scope_key="external"),
    ],
)
def test_canonical_integration_bindings_reject_untrusted_owner_dimensions(
    scope: StorageScope,
) -> None:
    with pytest.raises(ValueError):
        CanonicalIngressIdempotencyStore(
            repository=object(),  # type: ignore[arg-type]
            owner_scope=scope,
            clock=lambda: _NOW,
        )
    with pytest.raises(ValueError):
        CanonicalExternalSessionBindingStore(
            repository=object(),  # type: ignore[arg-type]
            owner_scope=scope,
        )


def test_canonical_integration_public_docstrings_follow_strict_contract() -> None:
    methods = (
        CanonicalIngressIdempotencyStore.__init__,
        CanonicalIngressIdempotencyStore.claim,
        CanonicalIngressIdempotencyStore.complete,
        CanonicalExternalSessionBindingStore.__init__,
        CanonicalExternalSessionBindingStore.get_or_create,
        CanonicalExternalSessionBindingStore.get,
        CanonicalInboundEventStore.__init__,
        CanonicalInboundEventStore.append,
        CanonicalSemanticEventStore.__init__,
        CanonicalSemanticEventStore.append,
        CanonicalSemanticEventStore.list_session,
        bind_canonical_integration_persistence,
    )
    for method in methods:
        docstring = getdoc(method)
        assert docstring is not None
        assert docstring.count("```python") == 2
        positions = [
            docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
        ]
        assert positions == sorted(positions)
