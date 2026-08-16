from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.services.sessions import SessionStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus, SessionKind
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex
from aethergraph.storage.artifacts.fs_cas import FSArtifactStore
from aethergraph.storage.runs.doc_store import DocRunStore
from aethergraph.storage.runs.inmen_store import InMemoryRunStore
from aethergraph.storage.runs.sqlite_run_store import SQLiteRunStore
from aethergraph.storage.sessions.doc_store import DocSessionStore
from aethergraph.storage.sessions.inmem_store import InMemorySessionStore
from aethergraph.storage.sessions.sqlite_session_store import SQLiteSessionStore

NOW = datetime(2026, 8, 15, 21, tzinfo=UTC)


class _MemoryDocStore:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, object]] = {}

    async def put(self, doc_id: str, doc: dict[str, object]) -> None:
        self.documents[doc_id] = copy.deepcopy(doc)

    async def get(self, doc_id: str) -> dict[str, object] | None:
        document = self.documents.get(doc_id)
        return copy.deepcopy(document) if document is not None else None

    async def delete(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)

    async def list(self) -> list[str]:
        return list(self.documents)


async def _exercise_artifact_identity(store: SessionStore) -> None:
    session = await store.create(
        session_id="session-1",
        kind=SessionKind.chat,
        user_id="user-1",
        org_id="org-1",
    )
    await store.record_artifact(
        session.session_id,
        occurrence_id="session-occurrence-1",
        created_at=NOW,
    )
    await store.record_artifact(
        session.session_id,
        occurrence_id="session-occurrence-1",
        created_at=NOW,
    )
    await store.record_artifact(
        session.session_id,
        occurrence_id="session-occurrence-2",
        created_at=NOW,
    )
    current = await store.get(session.session_id)
    assert current is not None
    assert current.artifact_count == 2
    assert current.last_artifact_at == NOW

    with pytest.raises(ValueError, match="identity conflicts"):
        await store.record_artifact(
            session.session_id,
            occurrence_id="session-occurrence-1",
            created_at=NOW + timedelta(seconds=1),
        )
    await store.record_artifact(
        "missing",
        occurrence_id="session-occurrence-missing",
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_in_memory_session_artifact_identity_is_idempotent() -> None:
    await _exercise_artifact_identity(InMemorySessionStore())


@pytest.mark.asyncio
async def test_document_session_artifact_identity_is_idempotent() -> None:
    await _exercise_artifact_identity(DocSessionStore(_MemoryDocStore()))


@pytest.mark.asyncio
async def test_sqlite_session_artifact_identity_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    await _exercise_artifact_identity(SQLiteSessionStore(str(path)))

    restarted = SQLiteSessionStore(str(path))
    await restarted.record_artifact(
        "session-1",
        occurrence_id="session-occurrence-1",
        created_at=NOW,
    )
    await restarted.record_artifact(
        "session-1",
        occurrence_id="session-occurrence-3",
        created_at=NOW,
    )
    current = await restarted.get("session-1")
    assert current is not None
    assert current.artifact_count == 3


async def _exercise_run_artifact_identity(store: RunStore) -> None:
    record = RunRecord(
        run_id="run-1",
        graph_id="graph-1",
        kind="taskgraph",
        status=RunStatus.running,
        started_at=NOW,
    )
    await store.create(record)
    await store.record_artifact(
        record.run_id,
        artifact_id="same-content",
        occurrence_id="run-occurrence-1",
        created_at=NOW,
    )
    await store.record_artifact(
        record.run_id,
        artifact_id="same-content",
        occurrence_id="run-occurrence-1",
        created_at=NOW,
    )
    await store.record_artifact(
        record.run_id,
        artifact_id="same-content",
        occurrence_id="run-occurrence-2",
        created_at=NOW,
    )
    current = await store.get(record.run_id)
    assert current is not None
    assert current.artifact_count == 2
    assert current.recent_artifact_ids == ["same-content", "same-content"]

    with pytest.raises(ValueError, match="identity conflicts"):
        await store.record_artifact(
            record.run_id,
            artifact_id="different-content",
            occurrence_id="run-occurrence-1",
            created_at=NOW,
        )


@pytest.mark.asyncio
async def test_in_memory_run_artifact_occurrence_identity_is_idempotent() -> None:
    await _exercise_run_artifact_identity(InMemoryRunStore())


@pytest.mark.asyncio
async def test_document_run_artifact_occurrence_identity_is_idempotent() -> None:
    await _exercise_run_artifact_identity(DocRunStore(_MemoryDocStore()))


@pytest.mark.asyncio
async def test_sqlite_run_artifact_occurrence_identity_survives_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runs.db"
    await _exercise_run_artifact_identity(SQLiteRunStore(str(path)))

    restarted = SQLiteRunStore(str(path))
    await restarted.record_artifact(
        "run-1",
        artifact_id="same-content",
        occurrence_id="run-occurrence-1",
        created_at=NOW,
    )
    current = await restarted.get("run-1")
    assert current is not None
    assert current.artifact_count == 2


@pytest.mark.asyncio
async def test_artifact_facade_forwards_one_occurrence_to_run_and_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = InMemoryRunStore()
    sessions = InMemorySessionStore()
    await runs.create(
        RunRecord(
            run_id="run-facade",
            graph_id="graph-1",
            kind="taskgraph",
            status=RunStatus.running,
            started_at=NOW,
        )
    )
    await sessions.create(
        session_id="session-facade",
        kind=SessionKind.chat,
        user_id="user-1",
        org_id="org-1",
    )
    monkeypatch.setattr(
        "aethergraph.services.artifacts.facade.current_services",
        lambda: SimpleNamespace(run_store=runs, session_store=sessions),
    )
    monkeypatch.setattr(
        "aethergraph.services.artifacts.facade.current_metering",
        lambda: None,
    )
    store = FSArtifactStore(str(tmp_path / "cas"))
    index = SqliteArtifactIndex(str(tmp_path / "artifact-index.db"))
    facade = ArtifactFacade(
        run_id="run-facade",
        graph_id="graph-1",
        node_id="node-1",
        tool_name="test-tool",
        tool_version="1",
        art_store=store,
        art_index=index,
        scope=Scope(
            run_id="run-facade",
            graph_id="graph-1",
            node_id="node-1",
            session_id="session-facade",
            memory_level="run",
        ),
    )
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("same content", encoding="utf-8")
    second_path.write_text("same content", encoding="utf-8")
    first = await facade.save_file(str(first_path), kind="text", cleanup=False)
    second = await facade.save_file(str(second_path), kind="text", cleanup=False)

    assert first.artifact_id == second.artifact_id
    assert first.occurrence_id != second.occurrence_id
    run = await runs.get("run-facade")
    session = await sessions.get("session-facade")
    assert run is not None and run.artifact_count == 2
    assert session is not None and session.artifact_count == 2


def test_session_artifact_contract_requires_exact_identity_and_doc_sections() -> None:
    parameter = inspect.signature(SessionStore.record_artifact).parameters["occurrence_id"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    docstring = inspect.getdoc(SessionStore.record_artifact) or ""
    sections = tuple(
        docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
    )
    assert sections == tuple(sorted(sections))

    run_parameters = inspect.signature(RunStore.record_artifact).parameters
    assert run_parameters["occurrence_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["occurrence_id"].default is inspect.Parameter.empty
