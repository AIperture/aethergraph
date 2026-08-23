from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
import httpx
import pytest

from aethergraph.api.v1.agent_endpoints import _stream_cursor, router
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
from aethergraph.core.runtime.run_types import SessionKind
from aethergraph.services.host.endpoint_credentials import EndpointCredentialRegistry
from aethergraph.services.integration import (
    IntegrationSessionResolution,
    ManifestRouteResolver,
    ResourceIngressPolicy,
)
from aethergraph.storage.contracts import (
    SessionKind as StorageSessionKind,
    SessionRecord,
    StorageScope,
)
from tests._canonical_storage_fakes import make_session_store
from tests._integration_fixtures import contract_compatibility

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
        release_compatibility=contract_compatibility(),
        integration_routes=(route,),
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


class _Sessions:
    def __init__(self) -> None:
        self.by_conversation = {}

    async def provision(
        self,
        *,
        route,
        external_identity,
        request_scope,
        build_id,
        binding_id,
        ag_session_id,
        now,
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
            return IntegrationSessionResolution(
                binding=binding,
                session=SessionRecord(
                    session_id=binding.ag_session_id,
                    kind=StorageSessionKind.CHAT,
                    scope=StorageScope(
                        org_id=request_scope.org_id,
                        user_id=request_scope.user_id,
                        session_id=binding.ag_session_id,
                    ),
                    revision=1,
                    created_at=now,
                    updated_at=now,
                ),
                session_created=True,
                binding_created=True,
            )
        return IntegrationSessionResolution(
            binding=binding,
            session=SessionRecord(
                session_id=binding.ag_session_id,
                kind=StorageSessionKind.CHAT,
                scope=StorageScope(
                    org_id=request_scope.org_id,
                    user_id=request_scope.user_id,
                    session_id=binding.ag_session_id,
                ),
                revision=1,
                created_at=binding.created_at,
                updated_at=now,
            ),
            session_created=False,
            binding_created=False,
        )

    async def get_binding(self, *, route, external_identity):
        return self.by_conversation.get(external_identity.conversation_id)


class _Ingress:
    def __init__(self, manifest, canonical_sessions) -> None:
        self.route_resolver = ManifestRouteResolver(manifest)
        self.resource_ingress = SimpleNamespace(policy=ResourceIngressPolicy())
        self.session_store = _Sessions()
        self.canonical_sessions = canonical_sessions
        self.calls = []

    async def provision_session(
        self,
        *,
        route_id,
        external_identity,
        request_scope,
        binding_id,
        ag_session_id,
        now,
        title=None,
    ):
        route = self.route_resolver.require(route_id)
        tenant_id = external_identity.tenant_id
        user_id = external_identity.user_id
        await self.canonical_sessions.create(
            session_id=ag_session_id,
            kind=SessionKind.chat,
            user_id=user_id,
            org_id=tenant_id,
            title=title,
            source=route.integration_kind.value,
            external_ref=f"agent-endpoint:{route.endpoint_id}",
        )
        return await self.session_store.provision(
            route=route,
            external_identity=external_identity,
            request_scope=request_scope,
            build_id="build-1",
            binding_id=binding_id,
            ag_session_id=ag_session_id,
            now=now,
        )

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


class _RunStore:
    def __init__(self) -> None:
        self.records = {}

    async def get(self, run_id):
        return self.records.get(run_id)


class _RunManager:
    def __init__(self) -> None:
        self.canceled = []

    async def cancel_run(self, run_id):
        self.canceled.append(run_id)


def _app() -> tuple[FastAPI, SimpleNamespace]:
    manifest = _manifest()
    session_store = make_session_store()
    ingress = _Ingress(manifest, session_store)
    container = SimpleNamespace(
        integration_ingress=ingress,
        host_manifest=manifest,
        semantic_events=SimpleNamespace(),
        session_store=session_store,
        run_store=_RunStore(),
        run_manager=_RunManager(),
    )
    app = FastAPI()
    app.state.container = container
    app.state.endpoint_credentials = EndpointCredentialRegistry.from_manifest(manifest)
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_identity] = lambda: RequestIdentity(
        user_id="user-1",
        org_id="tenant-1",
        mode="local",
    )
    return app, container


async def _authenticate(client: httpx.AsyncClient, app: FastAPI) -> None:
    credentials = app.state.endpoint_credentials.take_launch_credentials()
    token = credentials["support"]
    assert app.state.endpoint_credentials.take_launch_credentials() == {}
    response = await client.post(
        "/api/v1/agent-endpoints/support/authenticate",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.anyio
async def test_endpoint_session_and_ingress_use_manifest_route() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
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
    binding = container.integration_ingress.session_store.by_conversation[session_id]
    assert binding.ag_session_id == session_id


@pytest.mark.anyio
async def test_endpoint_multipart_upload_reaches_canonical_ingress() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-upload"},
        )
        accepted = await client.post(
            "/api/v1/agent-endpoints/support/ingress",
            data={
                "session_id": created.json()["session_id"],
                "idempotency_key": "turn-upload-1",
            },
            files={"files": ("brief.txt", b"exact contents", "text/plain")},
        )

    assert accepted.status_code == 200
    call = container.integration_ingress.calls[0]
    declared = call["envelope"].attachments
    verified = call["verified"].attachments
    assert len(declared) == len(verified) == 1
    assert declared[0].attachment_id == verified[0].attachment_id
    assert declared[0].attachment_id.startswith("upload-")
    assert declared[0].filename == "brief.txt"
    assert declared[0].content_type == "text/plain"
    assert declared[0].size_bytes == 14
    assert verified[0].data == b"exact contents"


@pytest.mark.anyio
async def test_endpoint_artifact_context_requires_only_canonical_identity() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-context"},
        )
        accepted = await client.post(
            "/api/v1/agent-endpoints/support/ingress",
            files={
                "session_id": (None, created.json()["session_id"]),
                "idempotency_key": (None, "turn-context-1"),
                "attachments_json": (None, '[{"artifact_id":"artifact-1"}]'),
            },
        )

    assert accepted.status_code == 200
    attachment = container.integration_ingress.calls[0]["envelope"].attachments[0]
    assert attachment.source_kind == "artifact"
    assert attachment.source_id == "artifact-1"


@pytest.mark.anyio
async def test_endpoint_rejects_unexpected_multipart_upload_field() -> None:
    app, _ = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-invalid-upload"},
        )
        response = await client.post(
            "/api/v1/agent-endpoints/support/ingress",
            data={
                "session_id": created.json()["session_id"],
                "idempotency_key": "turn-invalid-upload-1",
            },
            files={"file": ("brief.txt", b"contents", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "endpoint.ingress_body_invalid"


@pytest.mark.anyio
async def test_endpoint_session_metadata_is_scoped_to_route() -> None:
    app, _ = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        descriptor = await client.get("/api/v1/agent-endpoints/support")
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-metadata", "title": "Original"},
        )
        session_id = created.json()["session_id"]
        listed = await client.get("/api/v1/agent-endpoints/support/sessions")
        fetched = await client.get(f"/api/v1/agent-endpoints/support/sessions/{session_id}")
        updated = await client.patch(
            f"/api/v1/agent-endpoints/support/sessions/{session_id}",
            json={"title": "Renamed"},
        )
        deleted = await client.delete(f"/api/v1/agent-endpoints/support/sessions/{session_id}")

    assert descriptor.status_code == 200
    assert descriptor.json() == {"endpoint_id": "support", "entry_agent_id": "agent.support"}
    assert listed.status_code == 200
    assert [item["session_id"] for item in listed.json()["items"]] == [session_id]
    assert fetched.status_code == 200
    assert updated.json()["title"] == "Renamed"
    assert deleted.status_code == 204


@pytest.mark.anyio
async def test_endpoint_session_metadata_rejects_mismatched_durable_binding() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        first = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-first"},
        )
        second = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-second"},
        )
        first_id = first.json()["session_id"]
        second_id = second.json()["session_id"]
        first_binding = container.integration_ingress.session_store.by_conversation[first_id]
        container.integration_ingress.session_store.by_conversation[first_id] = (
            first_binding.model_copy(update={"ag_session_id": second_id})
        )

        fetched = await client.get(f"/api/v1/agent-endpoints/support/sessions/{first_id}")
        listed = await client.get("/api/v1/agent-endpoints/support/sessions")

    assert fetched.status_code == 404
    assert [item["session_id"] for item in listed.json()["items"]] == [second_id]


@pytest.mark.anyio
async def test_endpoint_ingress_rejects_agent_selection_fields() -> None:
    app, _ = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
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


@pytest.mark.anyio
async def test_endpoint_cancel_requires_turn_owned_by_bound_session() -> None:
    app, container = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _authenticate(client, app)
        created = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-cancel"},
        )
        session_id = created.json()["session_id"]
        container.run_store.records["run-owned"] = SimpleNamespace(session_id=session_id)
        container.run_store.records["run-other"] = SimpleNamespace(session_id="another-session")

        canceled = await client.post(
            f"/api/v1/agent-endpoints/support/sessions/{session_id}/cancel",
            json={"turn_id": "run-owned"},
        )
        rejected = await client.post(
            f"/api/v1/agent-endpoints/support/sessions/{session_id}/cancel",
            json={"turn_id": "run-other"},
        )

    assert canceled.status_code == 200
    assert canceled.json() == {
        "turn_id": "run-owned",
        "status": "cancellation_requested",
    }
    assert rejected.status_code == 404
    assert container.run_manager.canceled == ["run-owned"]


@pytest.mark.anyio
async def test_endpoint_rejects_local_identity_without_scoped_credential() -> None:
    app, _ = _app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/agent-endpoints/support/sessions",
            json={"idempotency_key": "browser-unauthenticated"},
        )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "endpoint.authentication_required"


def test_stream_cursor_advances_from_last_event_id() -> None:
    assert _stream_cursor(after_cursor=None, last_event_id=None) is None
    assert _stream_cursor(after_cursor=4, last_event_id=None) == 4
    assert _stream_cursor(after_cursor=None, last_event_id="7") == 7
    assert _stream_cursor(after_cursor=4, last_event_id="9") == 9
    assert _stream_cursor(after_cursor=9, last_event_id="4") == 9


def test_stream_cursor_rejects_invalid_last_event_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _stream_cursor(after_cursor=0, last_event_id="not-a-cursor")

    assert getattr(exc_info.value, "status_code", None) == 400
    assert exc_info.value.detail["code"] == "endpoint.last_event_id_invalid"
