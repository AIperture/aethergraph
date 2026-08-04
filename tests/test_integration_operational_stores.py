from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from aethergraph.contracts.integration import (
    ExternalIdentity,
    HostManifest,
    IngressEnvelope,
    IngressReceipt,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    OriginAddress,
    SemanticEventKind,
)
from aethergraph.services.integration import (
    IngressIdempotencyError,
    IntegrationRouteError,
    ManifestRouteResolver,
    SessionBindingError,
    SQLiteExternalSessionBindingStore,
    SQLiteIngressIdempotencyStore,
    VerifiedIntegrationContext,
)
from tests._integration_fixtures import contract_compatibility

_DIGEST = "a" * 64
_NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _capabilities() -> IntegrationCapabilities:
    return IntegrationCapabilities(
        event_kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
        streaming=False,
        interactions=True,
        attachments=True,
        cancellation=True,
    )


def _route(
    route_id: str = "route-slack",
    *,
    enabled: bool = True,
    session_scope: str = "conversation_thread",
    match_policy: IntegrationMatchPolicy | None = None,
) -> IntegrationRoute:
    return IntegrationRoute(
        route_id=route_id,
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        entry_agent_id="agent.support",
        enabled=enabled,
        match_policy=match_policy or IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope=session_scope),
        required_capabilities=_capabilities(),
    )


def _manifest(*routes: IntegrationRoute) -> HostManifest:
    return HostManifest(
        deployment_id="deployment-1",
        build_id="build-1",
        source_digest=_DIGEST,
        build_root="C:/workspace/build-1",
        entrypoint_module="compiled_support.runtime",
        entrypoint_symbol="register",
        graph_id="graph.support",
        entry_agent_id="agent.support",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        release_compatibility=contract_compatibility(),
        integration_routes=routes,
        logical_output_bindings={"primary": "origin"},
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


def _identity(
    *,
    conversation_id: str = "channel-C1",
    thread_id: str | None = "thread-1",
    user_id: str = "user-U1",
) -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id="workspace-T1",
        conversation_id=conversation_id,
        thread_id=thread_id,
        user_id=user_id,
    )


def _envelope(
    *,
    route_hint: str | None = "route-slack",
    text: str = "Hello",
    idempotency_key: str = "event-E1",
    external_event_id: str = "event-E1",
) -> IngressEnvelope:
    return IngressEnvelope(
        integration_id="slack-main",
        route_hint=route_hint,
        external_identity=_identity(),
        external_event_id=external_event_id,
        idempotency_key=idempotency_key,
        received_at=_NOW,
        text=text,
        origin_address=OriginAddress(
            channel_key="slack:team/T1:chan/C1:thread/thread-1",
            capability_profile_id="slack-v1",
        ),
    )


def _verified() -> VerifiedIntegrationContext:
    return VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="workspace-T1",
    )


def test_manifest_route_resolver_requires_one_exact_authenticated_route() -> None:
    route = _route(
        match_policy=IntegrationMatchPolicy(
            external_tenant_ids=("workspace-T1",),
            external_conversation_ids=("channel-C1",),
        )
    )
    resolver = ManifestRouteResolver(_manifest(route))

    resolved = resolver.resolve(verified=_verified(), envelope=_envelope())

    assert resolved is route

    wrong_tenant = VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="workspace-other",
    )
    with pytest.raises(IntegrationRouteError) as exc_info:
        resolver.resolve(verified=wrong_tenant, envelope=_envelope())
    assert exc_info.value.code == "integration.identity_mismatch"


def test_manifest_route_resolver_rejects_ambiguity_and_disabled_route() -> None:
    ambiguous = ManifestRouteResolver(_manifest(_route("route-a"), _route("route-b")))
    with pytest.raises(IntegrationRouteError) as exc_info:
        ambiguous.resolve(verified=_verified(), envelope=_envelope(route_hint=None))
    assert exc_info.value.code == "integration.route_ambiguous"

    disabled = ManifestRouteResolver(_manifest(_route(enabled=False)))
    with pytest.raises(IntegrationRouteError) as exc_info:
        disabled.resolve(verified=_verified(), envelope=_envelope())
    assert exc_info.value.code == "integration.route_disabled"


@pytest.mark.asyncio
async def test_sqlite_session_binding_creation_is_atomic_and_build_pinned(tmp_path) -> None:
    store = SQLiteExternalSessionBindingStore(tmp_path / "integration.db")
    route = _route()

    async def create(index: int):
        return await store.get_or_create(
            route=route,
            external_identity=_identity(user_id=f"user-{index}"),
            build_id="build-1",
            binding_id=f"binding-{index}",
            ag_session_id=f"session-{index}",
            now=_NOW + timedelta(seconds=index),
        )

    results = await asyncio.gather(*(create(index) for index in range(8)))

    assert sum(result.created for result in results) == 1
    assert len({result.binding.binding_id for result in results}) == 1
    assert len({result.binding.ag_session_id for result in results}) == 1
    restored = SQLiteExternalSessionBindingStore(tmp_path / "integration.db")
    binding = await restored.get(route=route, external_identity=_identity(user_id="another-user"))
    assert binding is not None
    assert binding.binding_id == results[0].binding.binding_id

    with pytest.raises(SessionBindingError) as exc_info:
        await store.get_or_create(
            route=route,
            external_identity=_identity(),
            build_id="build-2",
            binding_id="replacement-binding",
            ag_session_id="replacement-session",
            now=_NOW + timedelta(days=1),
        )
    assert exc_info.value.code == "integration.binding_build_mismatch"


@pytest.mark.asyncio
async def test_sqlite_session_binding_requires_policy_thread_identity(tmp_path) -> None:
    store = SQLiteExternalSessionBindingStore(tmp_path / "integration.db")

    with pytest.raises(SessionBindingError) as exc_info:
        await store.get_or_create(
            route=_route(),
            external_identity=_identity(thread_id=None),
            build_id="build-1",
            binding_id="binding-1",
            ag_session_id="session-1",
            now=_NOW,
        )
    assert exc_info.value.code == "integration.binding_thread_required"


@pytest.mark.asyncio
async def test_sqlite_idempotency_claims_once_and_replays_terminal_receipt(tmp_path) -> None:
    store = SQLiteIngressIdempotencyStore(tmp_path / "integration.db")
    envelope = _envelope()

    claims = await asyncio.gather(
        *(store.claim(deployment_id="deployment-1", envelope=envelope) for _ in range(8))
    )

    assert sum(claim.acquired for claim in claims) == 1
    assert sum(claim.pending for claim in claims) == 7

    receipt = IngressReceipt(
        accepted=True,
        duplicate=False,
        action="root_turn_started",
        deployment_id="deployment-1",
        route_id="route-slack",
        session_id="session-1",
        turn_id="turn-1",
        event_cursor=1,
    )
    await store.complete(
        deployment_id="deployment-1",
        envelope=envelope,
        receipt=receipt,
    )

    restored = SQLiteIngressIdempotencyStore(tmp_path / "integration.db")
    redelivered = envelope.model_copy(update={"received_at": _NOW + timedelta(minutes=1)})
    duplicate = await restored.claim(deployment_id="deployment-1", envelope=redelivered)
    assert duplicate.acquired is False
    assert duplicate.pending is False
    assert duplicate.receipt == receipt.model_copy(update={"duplicate": True})


@pytest.mark.asyncio
async def test_sqlite_idempotency_rejects_key_reuse_and_second_completion(tmp_path) -> None:
    store = SQLiteIngressIdempotencyStore(tmp_path / "integration.db")
    envelope = _envelope()
    await store.claim(deployment_id="deployment-1", envelope=envelope)

    with pytest.raises(IngressIdempotencyError) as exc_info:
        await store.claim(
            deployment_id="deployment-1",
            envelope=_envelope(text="Different content"),
        )
    assert exc_info.value.code == "integration.idempotency_conflict"

    with pytest.raises(IngressIdempotencyError) as exc_info:
        await store.claim(
            deployment_id="deployment-1",
            envelope=_envelope(idempotency_key="another-key"),
        )
    assert exc_info.value.code == "integration.idempotency_conflict"

    receipt = IngressReceipt(
        accepted=False,
        duplicate=False,
        action="rejected",
        deployment_id="deployment-1",
        rejection_code="integration.route_not_found",
    )
    await store.complete(
        deployment_id="deployment-1",
        envelope=envelope,
        receipt=receipt,
    )
    with pytest.raises(IngressIdempotencyError) as exc_info:
        await store.complete(
            deployment_id="deployment-1",
            envelope=envelope,
            receipt=receipt,
        )
    assert exc_info.value.code == "integration.idempotency_already_completed"
