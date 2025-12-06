from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# Adjust this import to match your actual module path
from aethergraph.api.v1 import stats as stats_module
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.metering import MeteringService


class FakeMetering(MeteringService):
    """
    Minimal fake MeteringService for endpoint tests.
    We only care about the read methods + presence of the write methods.
    """

    # --- record_* methods: no-op ---

    async def record_llm(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int | None = None,
    ) -> None:
        return None

    async def record_run(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        status: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        return None

    async def record_artifact(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        kind: str,
        bytes: int,
        pinned: bool = False,
    ) -> None:
        return None

    async def record_event(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        scope_id: str | None = None,
        kind: str,
    ) -> None:
        return None

    # --- read methods: return deterministic dummy data ---

    async def get_overview(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: str | None = None,
    ) -> dict[str, int]:
        # assert calls are wired correctly
        assert window == "24h"
        # You can also assert user/org if you want
        return {
            "llm_calls": 3,
            "llm_prompt_tokens": 120,
            "llm_completion_tokens": 45,
            "runs": 2,
            "runs_succeeded": 1,
            "runs_failed": 1,
            "artifacts": 4,
            "artifact_bytes": 2048,
            "events": 7,
        }

    async def get_llm_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: str | None = None,
    ) -> dict[str, dict[str, int]]:
        # Example: stats per model
        return {
            "gpt-4o-mini": {
                "calls": 2,
                "prompt_tokens": 80,
                "completion_tokens": 30,
            },
            "gpt-4.1": {
                "calls": 1,
                "prompt_tokens": 40,
                "completion_tokens": 15,
            },
        }

    async def get_graph_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: str | None = None,
    ) -> dict[str, dict[str, int]]:
        # Example: stats keyed by graph_id
        return {
            "graph.optimize_lens": {
                "runs": 2,
                "succeeded": 1,
                "failed": 1,
                "total_duration_s": 12,
            },
            "graph.etl": {
                "runs": 1,
                "succeeded": 1,
                "failed": 0,
                "total_duration_s": 5,
            },
        }

    async def get_artifact_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: str | None = None,
    ) -> dict[str, dict[str, int]]:
        # Example: stats keyed by artifact kind
        return {
            "json": {
                "count": 2,
                "bytes": 1024,
                "pinned_count": 1,
                "pinned_bytes": 256,
            },
            "image": {
                "count": 1,
                "bytes": 4096,
                "pinned_count": 0,
                "pinned_bytes": 0,
            },
        }

    async def get_memory_stats(
        self,
        *,
        scope_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: str | None = None,
    ) -> dict[str, dict[str, int]]:
        # Example: memory kinds, we only track count
        base = {
            "memory.user_msg": {"count": 5},
            "memory.tool_result": {"count": 2},
        }

        if scope_id is not None:
            # In a real impl you'd filter here; for testing we just return unchanged
            return base
        return base


class FakeContainer:
    """
    Lightweight container that exposes .metering like the real runtime container.
    """

    def __init__(self, metering: MeteringService):
        self.metering = metering


@pytest.fixture
def fake_metering() -> FakeMetering:
    return FakeMetering()


@pytest.fixture
def app(monkeypatch, fake_metering: FakeMetering) -> FastAPI:
    """
    Build a FastAPI app with the stats router and monkeypatch current_services
    to return a container that has .metering = fake_metering.
    """
    # Create container
    container = FakeContainer(fake_metering)

    # Monkeypatch the current_services symbol imported inside the stats module.
    # Note: stats_module.current_services was imported like:
    #   from aethergraph.core.runtime.runtime_services import current_services
    def fake_current_services():
        return container

    monkeypatch.setattr(stats_module, "current_services", fake_current_services)

    # Override get_identity dependency to a deterministic identity
    def override_get_identity():
        return RequestIdentity(user_id="user-1", org_id="org-1")

    app = FastAPI()
    app.include_router(stats_module.router)

    app.dependency_overrides[stats_module.get_identity] = override_get_identity

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_stats_overview(client: TestClient):
    resp = client.get("/stats/overview?window=24h")
    assert resp.status_code == 200

    data = resp.json()
    # Check shape & a couple of fields
    assert data["llm_calls"] == 3
    assert data["llm_prompt_tokens"] == 120
    assert data["runs"] == 2
    assert data["artifacts"] == 4
    assert data["events"] == 7


def test_graphs_stats_all(client: TestClient):
    resp = client.get("/stats/graphs?window=24h")
    assert resp.status_code == 200

    data = resp.json()
    print("🍏", data)
    # GraphStats is a root-map: { graph_id: {runs, succeeded, failed, total_duration_s}, ... }
    assert "graph.optimize_lens" in data
    assert "graph.etl" in data

    lens_stats = data["graph.optimize_lens"]
    assert lens_stats["runs"] == 2
    assert lens_stats["succeeded"] == 1
    assert lens_stats["failed"] == 1
    assert lens_stats["total_duration_s"] == 12


def test_graphs_stats_filtered(client: TestClient):
    resp = client.get("/stats/graphs?graph_id=graph.etl&window=24h")
    assert resp.status_code == 200

    data = resp.json()
    # Only one key should be present
    assert list(data.keys()) == ["graph.etl"]
    etl_stats = data["graph.etl"]
    assert etl_stats["runs"] == 1
    assert etl_stats["succeeded"] == 1
    assert etl_stats["failed"] == 0


def test_memory_stats(client: TestClient):
    resp = client.get("/stats/memory?window=24h")
    assert resp.status_code == 200

    data = resp.json()
    # MemoryStats is a root-map: { "memory.user_msg": {"count": N}, ... }
    assert "memory.user_msg" in data
    assert "memory.tool_result" in data

    assert data["memory.user_msg"]["count"] == 5
    assert data["memory.tool_result"]["count"] == 2


def test_artifacts_stats(client: TestClient):
    resp = client.get("/stats/artifacts?window=24h")
    assert resp.status_code == 200

    data = resp.json()
    # ArtifactStats is a root-map: { kind: {count, bytes, pinned_count, pinned_bytes}, ... }
    assert "json" in data
    assert "image" in data

    json_stats = data["json"]
    assert json_stats["count"] == 2
    assert json_stats["bytes"] == 1024
    assert json_stats["pinned_count"] == 1
    assert json_stats["pinned_bytes"] == 256


def test_stats_llm(client: TestClient):
    resp = client.get("/stats/llm?window=24h")
    assert resp.status_code == 200

    data = resp.json()
    # LLMStats is a root-map: { model_name: {calls, prompt_tokens, completion_tokens}, ... }
    assert "gpt-4o-mini" in data
    assert "gpt-4.1" in data

    mini_stats = data["gpt-4o-mini"]
    assert mini_stats["calls"] == 2
    assert mini_stats["prompt_tokens"] == 80
    assert mini_stats["completion_tokens"] == 30


def test_stats_overview_no_metering(monkeypatch):
    app = FastAPI()

    # Reload router without monkeypatching metering
    from aethergraph.api.v1 import stats as stats_mod

    app.include_router(stats_mod.router)

    def override_get_identity():
        return RequestIdentity(user_id="user-1", org_id="org-1")

    app.dependency_overrides[stats_mod.get_identity] = override_get_identity

    class EmptyContainer:
        pass

    def fake_current_services():
        return EmptyContainer()

    monkeypatch.setattr(stats_mod, "current_services", fake_current_services)

    client = TestClient(app)
    resp = client.get("/stats/overview")
    assert resp.status_code == 501  # or 503, depending on your code path
