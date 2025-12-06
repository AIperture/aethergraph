from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import artifacts as artifacts_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.storage.artifact_index import Artifact

# -----------------------------
# Fake services
# -----------------------------


class FakeArtifactIndex:
    """
    Simple in-memory implementation of an ArtifactIndex-like interface for testing.
    """

    def __init__(self, artifacts: list[Artifact]):
        # store by id
        self._artifacts: dict[str, Artifact] = {a.artifact_id: a for a in artifacts}
        self.pin_calls: list[tuple[str, bool]] = []
        self.occurrences: list[tuple[str, dict | None]] = []

    # --- required protocol methods --- #

    async def upsert(self, a: Artifact) -> None:
        self._artifacts[a.artifact_id] = a

    async def list_for_run(self, run_id: str) -> list[Artifact]:
        return [a for a in self._artifacts.values() if a.run_id == run_id]

    async def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, Any] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        results = list(self._artifacts.values())

        # kind filter
        if kind is not None:
            results = [a for a in results if a.kind == kind]

        # labels filter
        if labels:

            def artifact_matches(a: Artifact) -> bool:
                lbls = a.labels or {}
                for k, v in labels.items():
                    if k == "tags":
                        # allow filter tags as list[str]
                        filter_tags = v if isinstance(v, list) else [v]
                        art_tags_raw = lbls.get("tags", [])
                        if isinstance(art_tags_raw, str):
                            art_tags = [t.strip() for t in art_tags_raw.split(",") if t.strip()]
                        else:
                            art_tags = [str(t) for t in art_tags_raw]
                        # require all filter tags to be present
                        if not all(t in art_tags for t in filter_tags):
                            return False
                    else:
                        if lbls.get(k) != v:
                            return False
                return True

            results = [a for a in results if artifact_matches(a)]

        # metric sorting
        if metric and mode in ("max", "min"):

            def get_metric(a: Artifact) -> float:
                try:
                    return float(a.metrics.get(metric, 0.0))
                except Exception:
                    return 0.0

            reverse = mode == "max"
            results.sort(key=get_metric, reverse=reverse)

        if limit is not None:
            results = results[:limit]

        return results

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        filters: dict[str, Any] | None = None,
    ) -> Artifact | None:
        # delegate to search
        results = await self.search(
            kind=kind,
            labels=filters,
            metric=metric,
            mode=mode,
            limit=None,
        )
        if not results:
            return None
        return results[0]

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        self.pin_calls.append((artifact_id, pinned))
        a = self._artifacts.get(artifact_id)
        if a is not None:
            a.pinned = pinned

    async def record_occurrence(
        self,
        a: Artifact,
        extra_labels: dict | None = None,
    ) -> None:
        self.occurrences.append((a.artifact_id, extra_labels or {}))

    async def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)


class FakeArtifactStore:
    """
    Minimal artifact store-like class for testing content endpoint.
    """

    def __init__(self, data_by_uri: dict[str, bytes]):
        self._data_by_uri = data_by_uri
        self.base_uri = "fs://fake"

    async def load_artifact_bytes(self, uri: str) -> bytes:
        return self._data_by_uri[uri]


class FakeRunManager:
    """
    Minimal RunManager-like for endpoints that require run_manager presence.
    """

    async def get_record(self, run_id: str):
        # For local tests we don't enforce client scoping, so this is unused.
        return None


class FakeContainer:
    def __init__(
        self,
        index: FakeArtifactIndex,
        store: FakeArtifactStore,
        run_manager: FakeRunManager,
    ):
        self.artifact_index = index
        self.artifacts = store
        self.run_manager = run_manager


# -----------------------------
# Test fixture: FastAPI client
# -----------------------------


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    # Build some sample artifacts
    now_iso = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()

    a1 = Artifact(
        artifact_id="a1",
        uri="fs://fake/a1.txt",
        kind="file",
        bytes=10,
        sha256="sha-a1",
        mime="text/plain",
        run_id="run1",
        graph_id="g1",
        node_id="n1",
        tool_name="tool",
        tool_version="v1",
        created_at=now_iso,
        labels={"scope_id": "scope1", "tags": ["foo", "bar"], "extra": "x"},
        metrics={"score": 0.9},
        preview_uri=None,
        pinned=False,
    )

    a2 = Artifact(
        artifact_id="a2",
        uri="fs://fake/a2.png",
        kind="image",
        bytes=20,
        sha256="sha-a2",
        mime="image/png",
        run_id="run1",
        graph_id="g1",
        node_id="n2",
        tool_name="tool",
        tool_version="v1",
        created_at=now_iso,
        labels={"scope_id": "scope1", "tags": ["bar"], "extra": "y"},
        metrics={"score": 0.7},
        preview_uri=None,
        pinned=False,
    )

    a3 = Artifact(
        artifact_id="a3",
        uri="fs://fake/a3.txt",
        kind="file",
        bytes=30,
        sha256="sha-a3",
        mime="text/plain",
        run_id="run2",
        graph_id="g2",
        node_id="n3",
        tool_name="tool",
        tool_version="v1",
        created_at=now_iso,
        labels={"scope_id": "scope2", "tags": ["baz"]},
        metrics={"score": 1.2},
        preview_uri=None,
        pinned=False,
    )

    index = FakeArtifactIndex([a1, a2, a3])
    store = FakeArtifactStore(
        {
            "fs://fake/a1.txt": b"CONTENT-a1",
            "fs://fake/a2.png": b"\x89PNG...",
            "fs://fake/a3.txt": b"CONTENT-a3",
        }
    )
    run_manager = FakeRunManager()
    container = FakeContainer(index=index, store=store, run_manager=run_manager)

    # Patch current_services used inside artifacts API
    monkeypatch.setattr(
        "aethergraph.api.v1.artifacts.current_services",
        lambda: container,
    )

    # Build app and override identity dependency
    app = FastAPI()
    app.include_router(artifacts_api.router, prefix="/api/v1")

    from aethergraph.api.v1.artifacts import get_identity

    async def fake_get_identity():
        # Local mode, no client scoping
        return RequestIdentity(
            user_id="u1",
            org_id="o1",
            roles=["dev"],
            client_id=None,
            mode="local",
        )

    app.dependency_overrides[get_identity] = fake_get_identity

    client = TestClient(app)
    # attach fakes for inspection in tests
    client.fake_index = index
    client.fake_store = store
    client.fake_run_manager = run_manager
    return client


# -----------------------------
# Tests
# -----------------------------


def test_list_artifacts_basic(client: TestClient):
    resp = client.get("/api/v1/artifacts")
    assert resp.status_code == 200
    data = resp.json()

    assert "artifacts" in data
    arts = data["artifacts"]
    assert len(arts) == 3

    ids = {a["artifact_id"] for a in arts}
    assert ids == {"a1", "a2", "a3"}


def test_list_artifacts_with_filters(client: TestClient):
    # Filter: scope1 + kind=file + tags=foo
    resp = client.get(
        "/api/v1/artifacts",
        params={"scope_id": "scope1", "kind": "file", "tags": "foo"},
    )
    assert resp.status_code == 200
    data = resp.json()
    arts = data["artifacts"]

    # Only a1 matches: scope1, kind=file, tags include foo
    assert len(arts) == 1
    assert arts[0]["artifact_id"] == "a1"
    assert arts[0]["scope_id"] == "scope1"
    assert arts[0]["kind"] == "file"


def test_get_artifact_found(client: TestClient):
    resp = client.get("/api/v1/artifacts/a1")
    assert resp.status_code == 200
    art = resp.json()
    assert art["artifact_id"] == "a1"
    assert art["kind"] == "file"
    assert art["mime_type"] == "text/plain"
    assert art["scope_id"] == "scope1"


def test_get_artifact_not_found(client: TestClient):
    resp = client.get("/api/v1/artifacts/does-not-exist")
    assert resp.status_code == 404
    data = resp.json()
    assert data["detail"] == "Artifact does-not-exist not found"


def test_get_artifact_content(client: TestClient):
    resp = client.get("/api/v1/artifacts/a1/content")
    assert resp.status_code == 200
    assert resp.content == b"CONTENT-a1"
    # content-type from Response(media_type=...)
    assert resp.headers["content-type"].startswith("text/plain")
    # header is case-insensitive; router sets "X-AetherGraph-Artifact-Id"
    assert resp.headers["x-aethergraph-artifact-id"] == "a1"


def test_search_artifacts_general(client: TestClient):
    # Search for file artifacts in scope1 with extra="x"
    payload = {
        "query": "",
        "kind": "file",
        "scope_id": "scope1",
        "labels": {"extra": "x"},
        "limit": 10,
    }
    resp = client.post("/api/v1/artifacts/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    hits = data["hits"]
    assert len(hits) == 1
    hit = hits[0]
    assert hit["artifact"]["artifact_id"] == "a1"
    # score is 1.0 if no metric provided, that's fine


def test_search_artifacts_best_only(client: TestClient):
    # Best file artifact by metric 'score', mode=max -> a3 (score=1.2)
    payload = {
        "kind": "file",
        "metric": "score",
        "mode": "max",
        "best_only": True,
    }
    resp = client.post("/api/v1/artifacts/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    hits = data["hits"]

    assert len(hits) == 1
    hit = hits[0]
    assert hit["artifact"]["artifact_id"] == "a3"
    # score should equal metric from artifact
    assert hit["score"] == pytest.approx(1.2)


def test_list_run_artifacts(client: TestClient):
    # run1 has a1 and a2
    resp = client.get("/api/v1/runs/run1/artifacts")
    assert resp.status_code == 200
    data = resp.json()

    arts = data["artifacts"]
    ids = {a["artifact_id"] for a in arts}
    assert ids == {"a1", "a2"}


def test_pin_artifact(client: TestClient):
    # initially not pinned
    assert client.fake_index._artifacts["a1"].pinned is False

    resp = client.post("/api/v1/artifacts/a1/pin", json=True)
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"artifact_id": "a1", "pinned": True}

    # check index updated
    assert client.fake_index._artifacts["a1"].pinned is True
    assert ("a1", True) in client.fake_index.pin_calls
