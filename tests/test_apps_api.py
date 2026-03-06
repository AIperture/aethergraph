from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import apps as apps_api
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.services.registry.unified_registry import UnifiedRegistry


class FakeIdentity:
    def __init__(self, user_id: str | None = None, org_id: str | None = None, mode: str = "cloud"):
        self.user_id = user_id
        self.org_id = org_id
        self.mode = mode
        self.client_id = None
        self.roles = []


class FakeRunManager:
    def __init__(self):
        self.last_call = None

    async def submit_run(
        self,
        graph_id: str,
        *,
        inputs,
        run_id=None,
        tags=None,
        identity: FakeIdentity,
        session_id=None,
        origin=None,
        visibility=None,
        importance=None,
        agent_id=None,
        app_id=None,
        app_name=None,
        run_config=None,
    ):
        self.last_call = {
            "graph_id": graph_id,
            "app_id": app_id,
            "app_name": app_name,
            "inputs": inputs,
            "run_config": run_config,
            "user_id": identity.user_id,
            "org_id": identity.org_id,
        }
        return RunRecord(
            run_id=run_id or "run-app-1",
            graph_id=graph_id,
            kind="taskgraph",
            status=RunStatus.running,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            finished_at=None,
            tags=tags or [],
            user_id=identity.user_id,
            org_id=identity.org_id,
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    fake_rm = FakeRunManager()
    reg = UnifiedRegistry()
    reg.register(
        nspace="app",
        name="myapp",
        version="0.1.0",
        obj={"id": "myapp"},
        meta={"id": "myapp", "name": "My App", "graph_id": "g1"},
        tenant={"org_id": "o1", "user_id": "u1", "client_id": "browser-a"},
    )

    class FakeContainer:
        run_manager = fake_rm
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

    from aethergraph.api.v1.apps import get_identity

    async def fake_get_identity():
        return FakeIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[get_identity] = fake_get_identity
    tc = TestClient(app)
    tc.fake_rm = fake_rm
    return tc


def test_create_app_run_uses_app_metadata_graph(client: TestClient):
    resp = client.post(
        "/api/v1/apps/myapp/runs",
        json={
            "inputs": {"x": 1},
            "run_config": {"max_concurrency": 2},
            "appId": "ignored-by-app-endpoint",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["graph_id"] == "g1"
    assert data["run_id"] == "run-app-1"

    rm = client.fake_rm
    assert rm.last_call["graph_id"] == "g1"
    assert rm.last_call["app_id"] == "myapp"
    assert rm.last_call["app_name"] == "My App"
