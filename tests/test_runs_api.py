# tests/test_runs_api.py
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import runs as runs_api

# If your RateLimitSettings lives elsewhere, adjust this import:
from aethergraph.config.config import RateLimitSettings
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.services.rate_limit.inmem_rate_limit import SimpleRateLimiter


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
        session_id: str | None = None,
        origin: str | None = None,
        visibility: str | None = None,
        importance: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
    ):
        self.last_call = {
            "graph_id": graph_id,
            "inputs": inputs,
            "run_id": run_id,
            "tags": tags,
            "user_id": identity.user_id,
            "org_id": identity.org_id,
        }
        # For the HTTP API, submit_run returns just a RunRecord
        rec = RunRecord(
            run_id=run_id or "run-xyz",
            graph_id=graph_id,
            kind="taskgraph",
            status=RunStatus.succeeded,
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
            tags=tags or [],
            user_id=identity.user_id,
            org_id=identity.org_id,
        )
        return rec

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


class FakeMetering:
    """
    Minimal fake metering service that lets us control get_overview output.
    """

    def __init__(self, runs_count: int = 0):
        self.runs_count = runs_count

    async def get_overview(self, *, user_id=None, org_id=None, window: str = "1h", run_ids=None):
        # Only 'runs' field matters for the rate-limit dependency
        return {
            "runs": self.runs_count,
            "llm_calls": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "runs_succeeded": 0,
            "runs_failed": 0,
            "artifacts": 0,
            "artifact_bytes": 0,
            "events": 0,
        }


class FakeAuthz:
    async def authorize(self, identity, scope, action) -> None:
        """Always allow."""
        return


class FakeSettings:
    """
    Minimal settings object with a .rate_limit field so the dependency can read it.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_runs_per_window: int = 100,
        runs_window: str = "1h",
        burst_max_runs: int = 10,
        burst_window_seconds: int = 10,
    ):
        self.rate_limit = RateLimitSettings(
            enabled=enabled,
            max_concurrent_runs=8,
            max_runs_per_window=max_runs_per_window,
            runs_window=runs_window,
            burst_max_runs=burst_max_runs,
            burst_window_seconds=burst_window_seconds,
        )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """
    Baseline client with a FakeRunManager and no aggressive rate limits
    (large caps so normal tests don't trip them).
    """
    fake_rm = FakeRunManager()
    fake_meter = FakeMetering(runs_count=0)
    fake_settings = FakeSettings(
        enabled=True,
        max_runs_per_window=1000,  # effectively no window limit for baseline tests
        burst_max_runs=1000,  # effectively no burst limit
    )
    fake_authz = FakeAuthz()

    class FakeContainer:
        run_manager = fake_rm
        metering = fake_meter
        settings = fake_settings
        authz = fake_authz
        # No burst limiter for the baseline; the dependency will treat this as "no limiter"
        run_burst_limiter = None

    # Override current_services to return our fake container
    monkeypatch.setattr(
        "aethergraph.api.v1.runs.current_services",
        lambda: FakeContainer(),
    )
    monkeypatch.setattr(
        "aethergraph.api.v1.deps.current_services",
        lambda: FakeContainer(),
    )

    # Build app and override the identity dependency
    app = FastAPI()
    app.include_router(runs_api.router, prefix="/api/v1")

    from aethergraph.api.v1.runs import get_identity

    async def fake_get_identity():
        # Simulate a cloud-authenticated user
        return FakeIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[get_identity] = fake_get_identity

    client = TestClient(app)
    client.fake_rm = fake_rm
    client.fake_container = FakeContainer
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


def test_run_endpoint_window_rate_limit(monkeypatch):
    """
    If metering.get_overview returns runs >= max_runs_per_window,
    POST /runs should return 429.
    """
    fake_rm = FakeRunManager()
    fake_meter = FakeMetering(runs_count=50)  # pretend we've already used 50 runs
    fake_settings = FakeSettings(
        enabled=True,
        max_runs_per_window=10,  # very low cap to force 429
        runs_window="1h",
        burst_max_runs=1000,  # large so burst limit won't trigger here
    )
    fake_authz = FakeAuthz()

    # We do want a container instance type here
    class FakeContainer:
        run_manager = fake_rm
        metering = fake_meter
        settings = fake_settings
        run_burst_limiter = None
        authz = fake_authz

    monkeypatch.setattr(
        "aethergraph.api.v1.runs.current_services",
        lambda: FakeContainer(),
    )

    monkeypatch.setattr(
        "aethergraph.api.v1.deps.current_services",
        lambda: FakeContainer(),
    )

    app = FastAPI()
    app.include_router(runs_api.router, prefix="/api/v1")

    from aethergraph.api.v1.runs import get_identity

    async def fake_get_identity():
        return FakeIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[get_identity] = fake_get_identity

    client = TestClient(app)
    resp = client.post(
        "/api/v1/graphs/my-graph/runs",
        json={"inputs": {"x": 1}, "tags": ["t1"]},
    )
    assert resp.status_code == 429
    body = resp.json()
    assert "Run limit exceeded" in body["detail"]


def test_run_endpoint_burst_rate_limit(monkeypatch):
    """
    If the in-memory burst limiter is set to 1 event per window,
    the second POST /runs in quick succession should return 429.
    """
    fake_rm = FakeRunManager()
    fake_meter = FakeMetering(runs_count=0)  # no window pressure
    fake_settings = FakeSettings(
        enabled=True,
        max_runs_per_window=1000,  # high cap so window won't trigger
        runs_window="1h",
        burst_max_runs=1,
        burst_window_seconds=60,
    )

    burst_limiter = SimpleRateLimiter(
        max_events=fake_settings.rate_limit.burst_max_runs,
        window_seconds=fake_settings.rate_limit.burst_window_seconds,
    )
    fake_authz = FakeAuthz()

    class FakeContainer:
        run_manager = fake_rm
        metering = fake_meter
        settings = fake_settings
        run_burst_limiter = burst_limiter
        authz = fake_authz

    monkeypatch.setattr(
        "aethergraph.api.v1.runs.current_services",
        lambda: FakeContainer(),
    )
    monkeypatch.setattr(
        "aethergraph.api.v1.deps.current_services",
        lambda: FakeContainer(),
    )

    app = FastAPI()
    app.include_router(runs_api.router, prefix="/api/v1")

    from aethergraph.api.v1.runs import get_identity

    async def fake_get_identity():
        return FakeIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[get_identity] = fake_get_identity

    client = TestClient(app)

    # First call should succeed
    resp1 = client.post(
        "/api/v1/graphs/my-graph/runs",
        json={"inputs": {"x": 1}},
    )
    assert resp1.status_code == 200

    # Second call should hit the burst limiter
    resp2 = client.post(
        "/api/v1/graphs/my-graph/runs",
        json={"inputs": {"x": 2}},
    )
    assert resp2.status_code == 429
    body = resp2.json()
    assert "Too many runs started in a short period" in body["detail"]
