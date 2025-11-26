# tests/test_runs_api.py
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import runs as runs_api
from aethergraph.core.runtime.run_types import RunRecord, RunStatus


class FakeIdentity:
    def __init__(self, user_id: str | None = None, org_id: str | None = None):
        self.user_id = user_id
        self.org_id = org_id


class FakeRunManager:
    def __init__(self):
        self.last_call = None

    async def start_run(
        self,
        graph_id: str,
        *,
        inputs,
        run_id=None,
        tags=None,
        user_id=None,
        org_id=None,
    ):
        self.last_call = {
            "graph_id": graph_id,
            "inputs": inputs,
            "run_id": run_id,
            "tags": tags,
            "user_id": user_id,
            "org_id": org_id,
        }
        rec = RunRecord(
            run_id=run_id or "run-xyz",
            graph_id=graph_id,
            kind="taskgraph",
            status=RunStatus.succeeded,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tags=tags or [],
            user_id=user_id,
            org_id=org_id,
        )
        outputs = {"out": 123}
        has_waits = False
        continuations = []
        return rec, outputs, has_waits, continuations

    async def get_record(self, run_id: str):
        if run_id != "run-xyz":
            return None
        return RunRecord(
            run_id="run-xyz",
            graph_id="my-graph",
            kind="taskgraph",
            status=RunStatus.succeeded,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tags=["t1"],
            user_id="u1",
            org_id="o1",
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    fake_rm = FakeRunManager()

    class FakeContainer:
        run_manager = fake_rm

    # Override current_services to return our fake container
    monkeypatch.setattr(
        "aethergraph.api.v1.runs.current_services",
        lambda: FakeContainer(),
    )

    # Build app and override the dependency
    app = FastAPI()
    app.include_router(runs_api.router, prefix="/api/v1")

    from aethergraph.api.v1.runs import get_identity

    async def fake_get_identity():
        return FakeIdentity(user_id="u1", org_id="o1")

    # This is the key part: FastAPI's dependency_overrides
    app.dependency_overrides[get_identity] = fake_get_identity

    client = TestClient(app)
    client.fake_rm = fake_rm
    return client


def test_create_run_endpoint(client: TestClient):
    resp = client.post(
        "/api/v1/graphs/my-graph/runs",
        json={"inputs": {"x": 1}, "tags": ["t1"]},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["graph_id"] == "my-graph"
    assert data["run_id"] == "run-xyz"
    assert data["status"] == "succeeded"
    assert data["outputs"] == {"out": 123}
    assert data["has_waits"] is False

    rm = client.fake_rm
    # Now these should be 'u1'/'o1', not 'local'
    assert rm.last_call["user_id"] == "u1"
    assert rm.last_call["org_id"] == "o1"


def test_get_run_endpoint(client: TestClient):
    resp = client.get("/api/v1/runs/run-xyz")
    assert resp.status_code == 200
    data = resp.json()

    assert data["run_id"] == "run-xyz"
    assert data["graph_id"] == "my-graph"
    assert data["status"] == "succeeded"
    assert data["tags"] == ["t1"]
    assert data["user_id"] == "u1"
    assert data["org_id"] == "o1"
