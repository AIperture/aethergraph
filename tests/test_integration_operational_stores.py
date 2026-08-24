from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aethergraph.contracts.integration import (
    AgentInputV1,
    ExternalIdentity,
    HostManifest,
    IngressEnvelope,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    OriginAddress,
    SemanticEventKind,
)
from aethergraph.services.integration import (
    IntegrationRouteError,
    ManifestRouteResolver,
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
        input=AgentInputV1(
            input_id=external_event_id,
            kind="message",
            type="user.message",
            source="urn:test:slack",
            occurred_at=_NOW,
            payload={"text": text},
        ),
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
