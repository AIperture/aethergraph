from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import inspect
from pathlib import Path

import pytest

from aethergraph.config.storage_provider import StorageProviderSettings
from aethergraph.observability import (
    ObservabilityFacade,
    ObservabilityWorkspaceError,
    open_observability_workspace,
)
from aethergraph.services.memory.canonical_factory import CanonicalMemoryFacadeFactory
from aethergraph.storage.contracts import (
    ArtifactRecord,
    LLMCallDraft,
    LLMCallLifecycleStatus,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationScopeManagementRecord,
    ObservationSeverity,
    ObservationStatus,
    RunRecord,
    RunStatus,
    StorageOpenMode,
    StorageOpenRequest,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

NOW = datetime(2026, 8, 16, 20, tzinfo=UTC)
OWNER = StorageScope(project_id="project-1")


class _Clock:
    def now(self) -> datetime:
        return NOW


class _Secrets:
    async def resolve(self, reference: str) -> str:
        raise AssertionError(reference)


def _provider_and_request(root: Path):
    selection = StorageProviderSettings(provider="local.sqlite").to_selection()
    reference = selection.config["continuation_token_secret_ref"]
    assert isinstance(reference, str)
    provider = LocalStorageProvider(
        continuation_token_secret_ref=reference,
        continuation_token_secret=b"historical-workspace-test-secret-32-bytes",
    )
    request = StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=root.resolve(),
        owner_scope=OWNER,
        selection=selection,
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )
    return provider, request


@pytest.mark.asyncio
async def test_manifested_workspace_preserves_studio_and_engine_reader_boundary(
    tmp_path: Path,
) -> None:
    provider, request = _provider_and_request(tmp_path)
    bundle = provider.open(request)
    artifact_content = b"historical artifact bytes"

    async def artifact_chunks():
        yield artifact_content

    stored_blob = await bundle.blobs.put(OWNER, artifact_chunks())
    await bundle.artifacts.put(
        ArtifactRecord(
            artifact_id="artifact-1",
            content_hash=stored_blob.content_hash,
            hash_algorithm=stored_blob.hash_algorithm,
            size_bytes=stored_blob.size_bytes,
            media_type="application/octet-stream",
            kind="test",
            blob_locator=stored_blob.blob_locator,
            owner_scope=OWNER,
            created_at=NOW,
            original_filename="artifact.bin",
            provider_version=stored_blob.provider_version,
        )
    )
    run_scope = StorageScope(
        project_id="project-1",
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
    )
    await bundle.runs.create(
        RunRecord(
            run_id="run-1",
            graph_id="graph-1",
            kind="taskgraph",
            status=RunStatus.RUNNING,
            scope=run_scope,
            revision=1,
            started_at=NOW,
            metadata={
                "public_metadata": {"original_inputs": {"user_request": {"turn_id": "t-1"}}},
                "service_context": {
                    "origin": "app",
                    "visibility": "normal",
                    "importance": "normal",
                },
            },
        )
    )
    memory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=OWNER,
        clock=lambda: NOW,
        event_id_factory=lambda: "engine-1",
    ).for_public_execution(
        run_scope,
        logical_scope_id="run-1",
        provenance_scope=run_scope,
    )
    await memory.append_event(
        kind="agent_engine.decision",
        data={"turn_id": "t-1", "agent_instance_id": "agent-1"},
        text="Selected one Tool call",
        tags=["agent_engine", "turn:t-1"],
    )
    await bundle.observations.append_many(
        (
            ObservationDraft(
                observation_id="trace-1",
                category="service_operation",
                name="run",
                summary="runner/run end",
                occurred_at=NOW,
                scope=run_scope,
                status=ObservationStatus.OK,
                severity=ObservationSeverity.INFO,
                producer="runner",
                trace_id="trace-1",
                attributes={"operation": "run"},
            ),
        )
    )
    completed_call = LLMCallDraft(
        llm_call_id="call-1",
        observation=ObservationDraft(
            observation_id="llm-1",
            category="llm",
            name="chat",
            summary="openai/gpt-test chat",
            occurred_at=NOW,
            scope=run_scope,
            status=ObservationStatus.OK,
            severity=ObservationSeverity.INFO,
            producer="aethergraph.llm",
            trace_id="trace-1",
            attributes={
                "prompt_roles": ("user",),
                "prompt_chars": 42,
                "prompt_bytes": 42,
                "assembled_request_hash": "request-hash",
            },
        ),
        call_type="chat",
        provider="openai",
        model="gpt-test",
        capture_mode=ObservationCaptureMode.MANIFEST,
        prompt_manifest_id="manifest-1",
        captured_request={
            "messages": [{"role": "user", "content": "hello"}],
            "provider_request_args": {"temperature": 0},
        },
        captured_response={"text": "hi"},
    )
    started_call = replace(
        completed_call,
        observation=replace(completed_call.observation, status=ObservationStatus.PENDING),
        lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
        captured_response=None,
    )
    await bundle.observations.begin_llm_call(started_call)
    await bundle.observations.finish_llm_call("call-1", completed_call)
    await bundle.observations.compare_and_set_scope_management(
        ObservationScopeManagementRecord(
            scope_key="trace:trace-hidden",
            scope=OWNER,
            trace_id="trace-hidden",
            revision=1,
            updated_at=NOW,
            hidden=True,
        ),
        0,
    )
    await bundle.close()
    manifest_before = (tmp_path / "workspace.json").read_text(encoding="utf-8")

    facade = open_observability_workspace(tmp_path)
    traces = await facade.list_inspect_traces(run_id="run-1")
    runs = await facade.list_runs(limit=10_000, offset=0)
    session_runs = await facade.list_runs(
        limit=10_000,
        offset=0,
        session_id="session-1",
    )
    missing_session_runs = await facade.list_runs(
        limit=10_000,
        offset=0,
        session_id="missing-session",
    )
    engine_events = await facade.list_engine_events(run_id="run-1")
    suppressed = await facade.list_suppressed_scopes()
    manifest = await facade.hydrate_prompt_manifest("manifest-1")
    retained_artifact = await facade.read_artifact_bytes("artifact-1")
    missing_artifact = await facade.read_artifact_bytes("missing")
    await facade.close()
    await facade.close()

    assert [item.id for item in traces.items] == ["trace-1"]
    assert runs[0]["run_id"] == "run-1"
    assert [run["run_id"] for run in session_runs] == ["run-1"]
    assert missing_session_runs == []
    assert runs[0]["meta"]["original_inputs"]["user_request"]["turn_id"] == "t-1"
    assert engine_events == [
        {
            "event_id": "engine-1",
            "id": "engine-1",
            "ts": NOW.timestamp(),
            "run_id": "run-1",
            "session_id": "session-1",
            "kind": "agent_engine.decision",
            "text": "Selected one Tool call",
            "tags": ["agent_engine", "turn:t-1"],
            "data": {"turn_id": "t-1", "agent_instance_id": "agent-1"},
        }
    ]
    assert suppressed == {
        "session_id": set(),
        "run_id": set(),
        "trace_id": {"trace-hidden"},
    }
    assert manifest is not None
    assert manifest["manifest_id"] == "manifest-1"
    assert manifest["provider_request"]["messages"][0]["content"] == "hello"
    assert retained_artifact == artifact_content
    assert missing_artifact is None
    assert [part["content_kind"] for part in manifest["parts"]] == [
        "prompt_message",
        "provider_request_config",
    ]
    assert (tmp_path / "workspace.json").read_text(encoding="utf-8") == manifest_before


def test_workspace_opener_rejects_unmanifested_history_without_fallback(tmp_path: Path) -> None:
    (tmp_path / "events.db").write_bytes(b"legacy")

    with pytest.raises(ObservabilityWorkspaceError):
        open_observability_workspace(tmp_path)

    assert not (tmp_path / "workspace.json").exists()


def test_workspace_public_methods_keep_strict_docstrings() -> None:
    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    public_apis = [
        open_observability_workspace,
        *(
            member
            for name, member in inspect.getmembers(ObservabilityFacade)
            if not name.startswith("_")
        ),
    ]
    for public_api in public_apis:
        doc = inspect.getdoc(public_api) or ""
        assert [doc.index(section) for section in required] == sorted(
            doc.index(section) for section in required
        )
        assert doc.count("```python") >= 2


def test_workspace_opener_has_no_legacy_layout_or_concrete_store_dependency() -> None:
    source = inspect.getsource(inspect.getmodule(open_observability_workspace))

    for forbidden in (
        "events.db",
        "observability.db",
        "runs.db",
        "sqlite3",
        "SqliteEventLog",
        "SQLiteObservationStore",
        "_ReadOnlySQLiteRunStore",
    ):
        assert forbidden not in source
