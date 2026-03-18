from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import (
    apps as apps_api,
    artifacts as artifacts_api,
)
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.config.config import AppSettings
from aethergraph.server.app_factory import create_app
from aethergraph.services.auth.authn import DemoGrant
from aethergraph.services.registry.unified_registry import UnifiedRegistry


@pytest.fixture()
def auth_client(tmp_path) -> TestClient:
    cfg = AppSettings(
        workspace=str(tmp_path),
        deploy_mode="cloud",
        auth={"secret": "test-secret", "public_demo_fallback_enabled": True},
    )
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    with TestClient(app) as client:
        yield client


def _create_invite_and_redeem(client: TestClient, *, grant_id: str = "grant-1") -> dict:
    """Create an invite code on the server and redeem it, returning the auth/me body."""
    authn = client.app.state.container.authn
    grant = DemoGrant(
        grant_id=grant_id,
        org_id="org-demo",
        allowed_apps=["allowed-app"],
        allowed_agents=["allowed-agent"],
        client_label="Demo Client",
    )
    invite = authn.create_invite_code(grant)
    resp = client.post("/api/v1/auth/invite/redeem", json={"code": invite.code})
    assert resp.status_code == 200
    return resp.json()


def test_invite_redeem_sets_guest_session_cookie_and_auth_me(auth_client: TestClient) -> None:
    body = _create_invite_and_redeem(auth_client)
    assert body["mode"] == "demo"
    assert body["grant_id"] == "grant-1"
    assert body["catalog_scope"]["apps"] == ["allowed-app"]

    me = auth_client.get("/api/v1/auth/me")
    assert me.status_code == 200
    me_body = me.json()
    assert me_body["authenticated"] is True
    assert me_body["client_label"] == "Demo Client"
    assert me_body["user_id"].startswith("demo_guest:grant-1:")


def test_same_grant_creates_distinct_guest_users(auth_client: TestClient) -> None:
    authn = auth_client.app.state.container.authn
    grant = DemoGrant(grant_id="grant-shared", org_id="org-demo")
    sess_a = authn.create_demo_session(grant=grant)
    sess_b = authn.create_demo_session(grant=grant)
    assert sess_a.user_id != sess_b.user_id
    assert sess_a.org_id == sess_b.org_id == "org-demo"


def test_cloud_proxy_headers_resolve_identity(auth_client: TestClient) -> None:
    resp = auth_client.get(
        "/api/v1/whoami",
        headers={"X-User-ID": "u-proxy", "X-Org-ID": "o-proxy", "X-Roles": "admin,member"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "cloud"
    assert body["user_id"] == "u-proxy"
    assert body["org_id"] == "o-proxy"
    assert body["auth_source"] == "cloud_proxy_headers"


def test_public_demo_fallback_uses_browser_client_id(tmp_path) -> None:
    cfg = AppSettings(
        workspace=str(tmp_path),
        deploy_mode="demo",
        auth={"secret": "test-secret", "public_demo_fallback_enabled": True},
    )
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    with TestClient(app) as client:
        resp = client.get("/api/v1/whoami", headers={"X-Client-ID": "browser-123"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "demo"
        assert body["user_id"] == "demo:browser-123"
        assert body["auth_source"] == "public_demo_client_id"


def test_session_chat_history_requires_session_owner(auth_client: TestClient) -> None:
    authn = auth_client.app.state.container.authn
    grant_a = DemoGrant(grant_id="grant-a", org_id="org-demo")
    invite_a = authn.create_invite_code(grant_a)

    with TestClient(auth_client.app) as client_a:
        client_a.post("/api/v1/auth/invite/redeem", json={"code": invite_a.code})
        me_a = client_a.get("/api/v1/auth/me").json()

        session_store = auth_client.app.state.container.session_store
        assert session_store is not None
        created = __import__("asyncio").run(
            session_store.create(
                kind="chat",
                title="Private",
                external_ref=None,
                user_id=me_a["user_id"],
                org_id=me_a["org_id"],
                source="test",
            )
        )

    grant_b = DemoGrant(grant_id="grant-b", org_id="org-demo")
    invite_b = authn.create_invite_code(grant_b)

    with TestClient(auth_client.app) as client_b:
        client_b.post("/api/v1/auth/invite/redeem", json={"code": invite_b.code})
        resp = client_b.get(f"/api/v1/sessions/{created.session_id}/chat/events")
        assert resp.status_code == 403


def test_session_chat_websocket_rejects_other_guest(auth_client: TestClient) -> None:
    authn = auth_client.app.state.container.authn
    grant_a = DemoGrant(grant_id="grant-ws-a", org_id="org-demo")
    invite_a = authn.create_invite_code(grant_a)

    with TestClient(auth_client.app) as client_a:
        client_a.post("/api/v1/auth/invite/redeem", json={"code": invite_a.code})
        me_a = client_a.get("/api/v1/auth/me").json()
        session_store = auth_client.app.state.container.session_store
        created = __import__("asyncio").run(
            session_store.create(
                kind="chat",
                title="WS Private",
                external_ref=None,
                user_id=me_a["user_id"],
                org_id=me_a["org_id"],
                source="test",
            )
        )

    grant_b = DemoGrant(grant_id="grant-ws-b", org_id="org-demo")
    invite_b = authn.create_invite_code(grant_b)

    with TestClient(auth_client.app) as client_b:
        client_b.post("/api/v1/auth/invite/redeem", json={"code": invite_b.code})
        with (
            pytest.raises(Exception),  # noqa: B017
            client_b.websocket_connect(f"/api/v1/ws/sessions/{created.session_id}/chat"),
        ):
            pass


def test_catalog_scope_filters_apps(monkeypatch) -> None:
    reg = UnifiedRegistry()
    reg.register(
        nspace="app",
        name="allowed-app",
        version="0.1.0",
        obj={"id": "allowed-app"},
        meta={"id": "allowed-app", "graph_id": "g-allowed"},
        tenant={"org_id": "org-demo", "user_id": "demo-user"},
    )
    reg.register(
        nspace="app",
        name="blocked-app",
        version="0.1.0",
        obj={"id": "blocked-app"},
        meta={"id": "blocked-app", "graph_id": "g-blocked"},
        tenant={"org_id": "org-demo", "user_id": "demo-user"},
    )

    class FakeContainer:
        run_manager = None
        settings = None
        metering = None
        run_burst_limiter = None
        authz = None

    monkeypatch.setattr("aethergraph.api.v1.apps.current_services", lambda: FakeContainer())
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: FakeContainer())
    monkeypatch.setattr(
        "aethergraph.api.v1.registry_helpers.current_services", lambda: FakeContainer()
    )
    monkeypatch.setattr("aethergraph.api.v1.registry_helpers.current_registry", lambda: reg)

    app = FastAPI()
    app.include_router(apps_api.router, prefix="/api/v1")

    async def fake_identity():
        return RequestIdentity(
            user_id="demo-user",
            org_id="org-demo",
            mode="demo",
            catalog_scope={"apps": ["allowed-app"]},
        )

    app.dependency_overrides[apps_api.get_identity] = fake_identity
    client = TestClient(app)
    resp = client.get("/api/v1/apps")
    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()] == ["allowed-app"]


def test_artifact_content_enforces_identity(monkeypatch) -> None:
    artifact = SimpleNamespace(
        artifact_id="art-1",
        preview_uri=None,
        uri="file://artifact.bin",
        mime="text/plain",
        labels={"user_id": "owner-a", "org_id": "org-a"},
    )

    class FakeIndex:
        async def get(self, artifact_id: str):
            return artifact if artifact_id == "art-1" else None

    class FakeStore:
        async def load_artifact_bytes(self, uri: str):
            return b"hello"

    class FakeContainer:
        artifact_index = FakeIndex()
        artifacts = FakeStore()
        run_manager = object()

    monkeypatch.setattr("aethergraph.api.v1.artifacts.current_services", lambda: FakeContainer())
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: FakeContainer())

    app = FastAPI()
    app.include_router(artifacts_api.router, prefix="/api/v1")

    async def fake_identity():
        return RequestIdentity(user_id="owner-b", org_id="org-a", mode="demo")

    app.dependency_overrides[artifacts_api.get_identity] = fake_identity
    client = TestClient(app)
    resp = client.get("/api/v1/artifacts/art-1/content")
    assert resp.status_code == 404
