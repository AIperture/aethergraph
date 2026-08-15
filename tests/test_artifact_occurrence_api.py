from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import artifacts as artifacts_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.schemas.artifacts import ArtifactSearchRequest
from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.artifacts import PublicArtifactSearchHit
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex
from aethergraph.storage.artifacts.fs_cas import FSArtifactStore
from aethergraph.storage.contracts import ArtifactMetricOrder, Page, PageRequest, SearchMode


def _build_facade(
    *,
    run_id: str,
    node_id: str,
    session_id: str | None,
    store: FSArtifactStore,
    index: SqliteArtifactIndex,
) -> ArtifactFacade:
    scope = Scope(
        run_id=run_id,
        graph_id="graph-1",
        node_id=node_id,
        session_id=session_id,
        memory_level="run",
    )
    return ArtifactFacade(
        run_id=run_id,
        graph_id="graph-1",
        node_id=node_id,
        tool_name="test_tool",
        tool_version="0.1.0",
        art_store=store,
        art_index=index,
        scope=scope,
    )


def _write_file(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_sqlite_index_lists_artifact_occurrences_for_duplicate_content(tmp_path: Path) -> None:
    store = FSArtifactStore(str(tmp_path / "cas"))
    index = SqliteArtifactIndex(str(tmp_path / "artifact_index.db"))

    facade_a = _build_facade(
        run_id="run-1", node_id="node-a", session_id="sess-1", store=store, index=index
    )
    facade_b = _build_facade(
        run_id="run-1", node_id="node-b", session_id="sess-1", store=store, index=index
    )

    path_a = tmp_path / "alpha.txt"
    path_b = tmp_path / "beta.txt"
    _write_file(path_a, "identical payload")
    _write_file(path_b, "identical payload")

    async def _save() -> tuple[str, list, list, list]:
        art_a = await facade_a.save_file(str(path_a), kind="text", name="alpha.txt", cleanup=False)
        _art_b = await facade_b.save_file(str(path_b), kind="text", name="beta.txt", cleanup=False)
        run_rows = await index.list_occurrences_for_run("run-1")
        session_rows = await index.list_occurrences_for_session("sess-1")
        deduped = await index.search(labels={"run_id": "run-1"})
        return art_a.artifact_id, run_rows, session_rows, deduped

    import asyncio

    artifact_id, run_rows, session_rows, deduped = asyncio.run(_save())

    assert len(run_rows) == 2
    assert len(session_rows) == 2
    assert {row.artifact_id for row in run_rows} == {artifact_id}
    assert len({row.occurrence_id for row in run_rows}) == 2
    assert {row.labels.get("filename") for row in run_rows} == {"alpha.txt", "beta.txt"}
    assert {row.node_id for row in run_rows} == {"node-a", "node-b"}
    assert len(deduped) == 1


def test_run_and_session_artifact_endpoints_use_occurrences(tmp_path: Path, monkeypatch) -> None:
    store = FSArtifactStore(str(tmp_path / "cas"))
    index = SqliteArtifactIndex(str(tmp_path / "artifact_index.db"))

    facade_a = _build_facade(
        run_id="run-1", node_id="node-a", session_id="sess-1", store=store, index=index
    )
    facade_b = _build_facade(
        run_id="run-1", node_id="node-b", session_id="sess-1", store=store, index=index
    )

    path_a = tmp_path / "sample_a.txt"
    path_b = tmp_path / "sample_b.txt"
    _write_file(path_a, "same-bytes")
    _write_file(path_b, "same-bytes")

    async def _save() -> str:
        art_a = await facade_a.save_file(
            str(path_a), kind="text", name="sample_a.txt", cleanup=False
        )
        await facade_b.save_file(str(path_b), kind="text", name="sample_b.txt", cleanup=False)
        return art_a.artifact_id

    import asyncio

    artifact_id = asyncio.run(_save())

    class FakeContainer:
        artifact_index = index
        artifacts = store
        run_manager = object()

    monkeypatch.setattr("aethergraph.api.v1.artifacts.current_services", lambda: FakeContainer())
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: FakeContainer())

    app = FastAPI()
    app.include_router(artifacts_api.router, prefix="/api/v1")

    async def fake_identity() -> RequestIdentity:
        return RequestIdentity(mode="local")

    app.dependency_overrides[artifacts_api.get_identity] = fake_identity
    client = TestClient(app)

    run_resp = client.get("/api/v1/runs/run-1/artifacts")
    assert run_resp.status_code == 200
    run_payload = run_resp.json()
    assert len(run_payload["artifacts"]) == 2
    assert {item["artifact_id"] for item in run_payload["artifacts"]} == {artifact_id}
    assert len({item["occurrence_id"] for item in run_payload["artifacts"]}) == 2
    assert {item["filename"] for item in run_payload["artifacts"]} == {
        "sample_a.txt",
        "sample_b.txt",
    }

    session_resp = client.get("/api/v1/sessions/sess-1/artifacts")
    assert session_resp.status_code == 200
    session_payload = session_resp.json()
    assert len(session_payload["artifacts"]) == 2

    content_resp = client.get(f"/api/v1/artifacts/{artifact_id}/content")
    assert content_resp.status_code == 200
    assert content_resp.text == "same-bytes"

    global_resp = client.get("/api/v1/artifacts", params={"run_id": "run-1"})
    assert global_resp.status_code == 200
    global_payload = global_resp.json()
    assert len(global_payload["artifacts"]) == 1


def test_artifact_search_returns_frozen_hits_field(tmp_path: Path, monkeypatch) -> None:
    store = FSArtifactStore(str(tmp_path / "cas"))
    index = SqliteArtifactIndex(str(tmp_path / "artifact_index.db"))
    facade = _build_facade(
        run_id="run-1",
        node_id="node-1",
        session_id="sess-1",
        store=store,
        index=index,
    )
    source = tmp_path / "report.txt"
    _write_file(source, "ranked report")

    async def _save() -> str:
        artifact = await facade.save_file(
            str(source),
            kind="report",
            metrics={"quality": 0.9},
            cleanup=False,
        )
        return artifact.artifact_id

    import asyncio

    artifact_id = asyncio.run(_save())

    class FakeContainer:
        artifact_index = index

    monkeypatch.setattr("aethergraph.api.v1.artifacts.current_services", lambda: FakeContainer())
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: FakeContainer())

    app = FastAPI()
    app.include_router(artifacts_api.router, prefix="/api/v1")

    async def fake_identity() -> RequestIdentity:
        return RequestIdentity(mode="local")

    app.dependency_overrides[artifacts_api.get_identity] = fake_identity
    client = TestClient(app)

    response = client.post(
        "/api/v1/artifacts/search",
        json={
            "kind": "report",
            "metric": "quality",
            "mode": "max",
            "best_only": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"hits"}
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["score"] == 0.9
    assert payload["hits"][0]["artifact"]["artifact_id"] == artifact_id
    assert payload["hits"][0]["artifact"]["kind"] == "report"


def test_artifact_search_without_index_returns_empty_hits(monkeypatch) -> None:
    class FakeContainer:
        artifact_index = None

    monkeypatch.setattr("aethergraph.api.v1.artifacts.current_services", lambda: FakeContainer())
    monkeypatch.setattr("aethergraph.api.v1.deps.current_services", lambda: FakeContainer())

    app = FastAPI()
    app.include_router(artifacts_api.router, prefix="/api/v1")

    async def fake_identity() -> RequestIdentity:
        return RequestIdentity(mode="local")

    app.dependency_overrides[artifacts_api.get_identity] = fake_identity
    response = TestClient(app).post("/api/v1/artifacts/search", json={})

    assert response.status_code == 200
    assert response.json() == {"hits": []}


class _CanonicalSearchFacade:
    def __init__(self) -> None:
        self.text_call: dict[str, object] | None = None
        self.structural_call: dict[str, object] | None = None

    async def search_public_artifacts(self, **kwargs) -> tuple[PublicArtifactSearchHit, ...]:
        self.text_call = kwargs
        return (
            PublicArtifactSearchHit(
                artifact=_canonical_public_artifact(),
                score=0.75,
                mode=SearchMode.LEXICAL,
            ),
        )

    async def query_public_artifacts(self, page, **kwargs) -> Page[Artifact]:
        self.structural_call = {"page": page, **kwargs}
        return Page(items=(_canonical_public_artifact(),), next_cursor=None)


def _canonical_public_artifact() -> Artifact:
    return Artifact(
        artifact_id="artifact-1",
        kind="report",
        mime="text/plain",
        bytes=10,
        created_at="2026-08-15T00:00:00+00:00",
        labels={"scope_id": "logical-scope", "tags": ["final", "reviewed"]},
        metrics={"quality": 0.9},
    )


@pytest.mark.asyncio
async def test_canonical_artifact_search_mapper_uses_exact_lexical_mode() -> None:
    facade = _CanonicalSearchFacade()

    response = await artifacts_api._search_canonical_artifacts(
        ArtifactSearchRequest(
            query="  migration report  ",
            scope_id=" logical-scope ",
            kind=" report ",
            tags=[" reviewed ", "final"],
            labels={"stage": "published"},
            limit=7,
        ),
        facade,  # type: ignore[arg-type]
    )

    assert facade.text_call == {
        "query": "migration report",
        "mode": SearchMode.LEXICAL,
        "top_k": 7,
        "tags": ("final", "reviewed"),
        "metadata": {
            "kind": "report",
            "scope_id": "logical-scope",
            "stage": "published",
        },
    }
    assert facade.structural_call is None
    assert response.hits[0].score == 0.75
    assert response.hits[0].artifact is not None
    assert response.hits[0].artifact.artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_canonical_artifact_search_mapper_uses_indexed_metric_ranking() -> None:
    facade = _CanonicalSearchFacade()

    response = await artifacts_api._search_canonical_artifacts(
        ArtifactSearchRequest(
            kind="report",
            tags=["final"],
            labels={"stage": "published"},
            metric="quality",
            mode="max",
            limit=20,
            best_only=True,
        ),
        facade,  # type: ignore[arg-type]
    )

    assert facade.text_call is None
    assert facade.structural_call == {
        "page": PageRequest(limit=1),
        "kind": "report",
        "tags": ("final",),
        "labels": {"stage": "published"},
        "metric": "quality",
        "metric_order": ArtifactMetricOrder.MAXIMUM,
    }
    assert response.hits[0].score == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            {"query": "report", "metric": "quality", "mode": "max"},
            "text search cannot be combined",
        ),
        ({"metric": "quality"}, "metric and mode must be supplied together"),
        ({"best_only": True}, "best_only requires metric and mode"),
        ({"tags": ["final", " final "]}, "tags must not contain duplicates"),
        ({"labels": {"app_id": "legacy"}}, "app_id"),
        ({"labels": {"tags": ["final"]}}, "tags"),
    ],
)
async def test_canonical_artifact_search_mapper_rejects_ambiguous_requests(
    payload: dict[str, object],
    detail: str,
) -> None:
    facade = _CanonicalSearchFacade()
    with pytest.raises(HTTPException) as failure:
        await artifacts_api._search_canonical_artifacts(
            ArtifactSearchRequest(**payload),
            facade,  # type: ignore[arg-type]
        )

    assert getattr(failure.value, "status_code", None) == 422
    assert detail in str(getattr(failure.value, "detail", ""))
    assert facade.text_call is None
    assert facade.structural_call is None
