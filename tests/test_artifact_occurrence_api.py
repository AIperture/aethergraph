from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aethergraph.api.v1 import artifacts as artifacts_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex
from aethergraph.storage.artifacts.fs_cas import FSArtifactStore


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
