from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import registry as registry_api
from aethergraph.services.registry.registration_service import RegistrationResult


class FakeIdentity:
    def __init__(self, user_id: str | None = None, org_id: str | None = None, mode: str = "cloud"):
        self.user_id = user_id
        self.org_id = org_id
        self.mode = mode
        self.client_id = None
        self.roles = []


class FakeRegistryFacade:
    registration_service = object()

    async def register_by_file(self, path: str, **kwargs):
        _ = kwargs
        return RegistrationResult(
            success=True,
            source_kind="file",
            source_ref=path,
            filename="demo.py",
            sha256="abc",
            graph_name="demo_graph",
            app_id="demo_app",
            agent_id=None,
            version="0.1.0",
            entry_id="entry-1",
            errors=[],
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        "aethergraph.api.v1.registry.scoped_registry",
        lambda identity: FakeRegistryFacade(),
    )
    app = FastAPI()
    app.include_router(registry_api.router, prefix="/api/v1")

    from aethergraph.api.v1.registry import get_identity

    async def fake_get_identity():
        return FakeIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[get_identity] = fake_get_identity
    return TestClient(app)


def test_registry_register_by_file(client: TestClient):
    resp = client.post(
        "/api/v1/registry/register",
        json={
            "source": "file",
            "path": "/tmp/demo.py",
            "app_config": {"id": "demo_app"},
            "persist": True,
            "strict": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["graph_name"] == "demo_graph"
    assert body["app_id"] == "demo_app"
