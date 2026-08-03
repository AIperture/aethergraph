from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest

from aethergraph.api.v1.agent_endpoints import router
from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.contracts.integration import (
    ExternalSessionBinding,
    HostManifest,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    SemanticEventKind,
)
from aethergraph.services.integration import BindingResolution, ManifestRouteResolver
from aethergraph.storage.sessions.inmem_store import InMemorySessionStore

_DIGEST = "a" * 64


def _manifest() -> HostManifest:
    route = IntegrationRoute(
        route_id="route-endpoint",
        endpoint_id="support",
        integration_id="endpoint-support",
        integration_kind=IntegrationKind.AGENT_ENDPOINT,
        entry_agent_id="agent.support",
        enabled=True,
        match_policy=IntegrationMatchPolicy(external_tenant_ids=("tenant-1",)),
        session_policy=IntegrationSessionPolicy(scope="conversation_user"),
        required_capabilities=IntegrationCapabilities(
            event_kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
            streaming=True,
            interactions=True,
            attachments=True,
            cancellation=True,
        ),
    )
    return HostManifest(
        deployment_id="deployment-1",
        build_id="build-1",
        source_digest=_DIGEST,
        build_root="C:/compiled/build-1",
        entrypoint_module="compiled.runtime",
        entrypoint_symbol="register",
        graph_id="graph.support",
        entry_agent_id="agent.support",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        integration_routes=(route,),
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


class _Bindings:
    def __init__(self) -> None:
        self.by_conversation = {}

    async def get_or_create(
        self, *, route, external_identity, build_id, binding_id, ag_session_id, now
    ):
        binding = self.by_conversation.get(external_identity.conversation_id)
        if binding is None:
            binding = ExternalSessionBinding(
                binding_id=binding_id,
                route_id=route.route_id,
                external_identity=external_identity,
                ag_session_id=ag_session_id,
                build_id=build_id,
                created_at=now,
                last_seen_at=now,
            )
            self.by_conversation[external_identity.conversation_id] = binding
            return BindingResolution(binding=binding, created=True)
        return BindingResolution(binding=binding, created=False)

    async def get(self, *, route, external_identity):
        return self.by_conversation.get(external_identity.conversation_id)


class _Ingress:
    def __init__(self, manifest) -> None:
        self.route_resolver = ManifestRouteResolver(manifest)
        self.binding_store = _Bindings()
        self.calls = []

    async def accept(self, *, verified, envelope):
        self.calls.append({"verified": verified, "envelope": envelope})
        return {
            "accepted": True,
            "duplicate": False,
            "action": "root_turn_started",
            "deployment_id": "deployment-1",
            "route_id": "route-endpoint",
            "session_id": "internal-session",
            "turn_id": "run-1",
            "event_cursor": 1,
        }


def _app() -> tuple[FastAPI, SimpleNamespace]:
    manifest = _manifest()
    ingress = _Ingress(manifest)
    container = SimpleNamespace(
        integration_ingress=ingress,
        host_manifest=manifest,
        semantic_events=SimpleNamespace(),
        session_store=InMemorySessionStore(),
    )
    app = FastAPI()
    app.state.container = container
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_identity] = lambda: RequestIdentity(
        user_id="user-1",
        org_id="tenant-1",
        mode="local",
    )
    return app, container


@pytest.mark.anyio
async def test_endpoint_session_and_ingress_use_manifest_route() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-1"},
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        accepted = await client.post(
            "/api/v1/agent-endpoints/support/ingress",
            json={
                "session_id": session_id,
                "idempotency_key": "turn-1",
                "text": "Hello",
            },
        )

    assert accepted.status_code == 200
    call = container.integration_ingress.calls[0]
    assert call["verified"].integration_id == "endpoint-support"
    assert call["verified"].integration_kind is IntegrationKind.AGENT_ENDPOINT
    assert call["envelope"].endpoint_id == "support"
    assert call["envelope"].external_identity.conversation_id == session_id
    assert call["envelope"].origin_address.channel_key.startswith("endpoint:support:session/")
    stored = await container.session_store.get(session_id)
    assert stored is not None
    assert stored.external_ref == "agent-endpoint:support"
    binding = container.integration_ingress.binding_store.by_conversation[session_id]
    assert binding.ag_session_id == session_id


@pytest.mark.anyio
async def test_endpoint_ingress_rejects_agent_selection_fields() -> None:
    app, _ = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-1"},
        )
        response = await client.post(
            "/api/v1/agent-endpoints/support/ingress",
            json={
                "session_id": created.json()["session_id"],
                "idempotency_key": "turn-1",
                "text": "Hello",
                "agent_id": "agent.other",
            },
        )

    assert response.status_code == 400
    assert "agent.other" not in str(response.json())
