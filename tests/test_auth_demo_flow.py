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
from aethergraph.services.auth.authn import AuthnService, DemoGrant
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.storage.kv.sqlite_kv_sync import SQLiteKVSync


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


# ---- Persistence tests ---------------------------------------------------


def test_invite_code_survives_service_restart(tmp_path) -> None:
    """Invite codes persisted to SQLite survive AuthnService restart."""
    db_path = str(tmp_path / "auth_kv.db")
    grant_store = SQLiteKVSync(db_path, prefix="grant:")
    invite_store = SQLiteKVSync(db_path, prefix="invite:")

    authn1 = AuthnService(
        secret="test-secret",
        grant_store=grant_store,
        invite_store=invite_store,
    )
    grant = DemoGrant(grant_id="persist-grant", org_id="org-persist")
    invite = authn1.create_invite_code(grant, code="DEMO-PERSIST-TEST")
    assert invite.code == "DEMO-PERSIST-TEST"

    # Simulate server restart: new AuthnService, same stores
    grant_store2 = SQLiteKVSync(db_path, prefix="grant:")
    invite_store2 = SQLiteKVSync(db_path, prefix="invite:")
    authn2 = AuthnService(
        secret="test-secret",
        grant_store=grant_store2,
        invite_store=invite_store2,
    )
    authn2.load_persisted()

    # Redeem the persisted invite code on the new instance
    sess = authn2.redeem_invite_code("DEMO-PERSIST-TEST")
    assert sess.org_id == "org-persist"
    assert sess.user_id.startswith("demo_guest:persist-grant:")


def test_invite_use_count_persists(tmp_path) -> None:
    """Incremented use count is persisted to the store."""
    db_path = str(tmp_path / "auth_kv.db")
    grant_store = SQLiteKVSync(db_path, prefix="grant:")
    invite_store = SQLiteKVSync(db_path, prefix="invite:")

    authn = AuthnService(
        secret="test-secret",
        grant_store=grant_store,
        invite_store=invite_store,
    )
    grant = DemoGrant(grant_id="uses-grant", org_id="org-uses")
    authn.create_invite_code(grant, max_uses=3, code="DEMO-USES")

    authn.redeem_invite_code("DEMO-USES")
    authn.redeem_invite_code("DEMO-USES")

    # Restart and verify uses count
    grant_store2 = SQLiteKVSync(db_path, prefix="grant:")
    invite_store2 = SQLiteKVSync(db_path, prefix="invite:")
    authn2 = AuthnService(
        secret="test-secret",
        grant_store=grant_store2,
        invite_store=invite_store2,
    )
    authn2.load_persisted()

    # Should allow one more (uses=2, max=3)
    authn2.redeem_invite_code("DEMO-USES")

    # Should reject (uses=3, max=3)
    with pytest.raises(ValueError, match="usage limit"):
        authn2.redeem_invite_code("DEMO-USES")


def test_custom_invite_code_collision_rejected(tmp_path) -> None:
    """Creating a duplicate custom code raises ValueError."""
    db_path = str(tmp_path / "auth_kv.db")
    grant_store = SQLiteKVSync(db_path, prefix="grant:")
    invite_store = SQLiteKVSync(db_path, prefix="invite:")

    authn = AuthnService(
        secret="test-secret",
        grant_store=grant_store,
        invite_store=invite_store,
    )
    grant = DemoGrant(grant_id="dup-grant", org_id="org-dup")
    authn.create_invite_code(grant, code="DEMO-UNIQUE")

    with pytest.raises(ValueError, match="already exists"):
        authn.create_invite_code(grant, code="DEMO-UNIQUE")


def test_admin_api_key_blocks_unauthorized(tmp_path) -> None:
    """POST /invite/create is rejected when admin_api_key is set but not provided."""
    cfg = AppSettings(
        workspace=str(tmp_path),
        deploy_mode="cloud",
        auth={
            "secret": "test-secret",
            "admin_api_key": "my-admin-key",
        },
    )
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    with TestClient(app) as client:
        # No key → rejected
        resp = client.post(
            "/api/v1/auth/invite/create",
            json={
                "grant_id": "g1",
                "org_id": "o1",
            },
        )
        assert resp.status_code == 403

        # Wrong key → rejected
        resp = client.post(
            "/api/v1/auth/invite/create",
            json={"grant_id": "g1", "org_id": "o1"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

        # Correct key → allowed
        resp = client.post(
            "/api/v1/auth/invite/create",
            json={"grant_id": "g1", "org_id": "o1"},
            headers={"Authorization": "Bearer my-admin-key"},
        )
        assert resp.status_code == 200


def test_no_admin_key_allows_open_access(auth_client: TestClient) -> None:
    """When admin_api_key is not configured, /invite/create is open."""
    resp = auth_client.post(
        "/api/v1/auth/invite/create",
        json={
            "grant_id": "open-grant",
            "org_id": "org-open",
        },
    )
    assert resp.status_code == 200
