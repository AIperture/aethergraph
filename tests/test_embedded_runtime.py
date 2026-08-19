from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.run_types import RunOrigin, RunRecord, RunStatus
from aethergraph.runtime import (
    EmbeddedRuntime,
    RuntimeArtifactScope,
    RuntimeIdentity,
    RuntimeOpenRequest,
    RuntimeRunRequest,
    open_embedded_runtime,
)


class _RunManager:
    def __init__(self) -> None:
        self.submitted = None

    async def submit_run(self, graph_id, **kwargs):
        self.submitted = (graph_id, kwargs)
        return RunRecord(
            run_id=kwargs["run_id"] or "generated-run",
            graph_id=graph_id,
            kind="graphfn",
            status=RunStatus.pending,
            started_at=datetime.now(UTC),
            tags=list(kwargs["tags"]),
            session_id=kwargs["session_id"],
            origin=kwargs["origin"],
            meta={"accepted": True},
        )


class _RuntimeOutput:
    def __init__(self, rows_by_run):
        self.rows_by_run = rows_by_run

    async def query(self, query):
        records = []
        for row in self.rows_by_run.get(query.scope.run_id, ()):
            cursor = row["_row_id"]
            records.append(
                SimpleNamespace(
                    output_id=f"output-{cursor}",
                    delivery_cursor=cursor,
                    cursor=str(cursor),
                    scope=SimpleNamespace(
                        session_id=None,
                        run_id=query.scope.run_id,
                        graph_id=None,
                        node_id="node-1",
                    ),
                    tool_name=None,
                    tags=(),
                    execution_id=f"execution-{cursor}",
                    stream=SimpleNamespace(value="stdout"),
                    text=row["kind"],
                    sequence=cursor,
                    partial=False,
                    truncated=False,
                    source="runtime",
                    eof=False,
                )
            )
        return SimpleNamespace(items=tuple(records), next_cursor=None)


def _container(**overrides):
    async def start_storage():
        return None

    async def close_storage():
        return None

    values = {
        "channels": SimpleNamespace(),
        "cont_store": SimpleNamespace(),
        "run_manager": _RunManager(),
        "run_result_store": None,
        "state_store": SimpleNamespace(),
        "storage_services": SimpleNamespace(runtime_output=_RuntimeOutput({})),
        "observability": None,
        "resume_router": SimpleNamespace(),
        "ext_services": {},
        "start_storage": start_storage,
        "close_storage": close_storage,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_open_embedded_runtime_applies_extensions_before_return(monkeypatch, tmp_path):
    container = _container(observability=SimpleNamespace())
    observed = {}

    def _build_default_container(*, root, cfg, channel_adapters, **storage):
        del storage
        observed.update(root=root, cfg=cfg, channel_adapters=channel_adapters)
        return container

    monkeypatch.setattr(
        "aethergraph.runtime.embedded.build_default_container",
        _build_default_container,
    )
    settings = SimpleNamespace()
    runtime = open_embedded_runtime(
        RuntimeOpenRequest(
            root=tmp_path,
            settings=settings,
            channel_adapters={"custom": object()},
            extensions={"studio.resource_provider": "provider"},
        )
    )

    assert isinstance(runtime, EmbeddedRuntime)
    assert observed == {
        "root": str(tmp_path),
        "cfg": settings,
        "channel_adapters": {"custom": observed["channel_adapters"]["custom"]},
    }
    assert container.ext_services == {"studio.resource_provider": "provider"}


def test_open_embedded_runtime_rejects_missing_required_service(monkeypatch, tmp_path):
    container = _container(run_manager=None, observability=SimpleNamespace())
    monkeypatch.setattr(
        "aethergraph.runtime.embedded.build_default_container",
        lambda **kwargs: container,
    )

    with pytest.raises(RuntimeError, match="run_manager"):
        open_embedded_runtime(RuntimeOpenRequest(root=tmp_path, settings=SimpleNamespace()))


@pytest.mark.asyncio
async def test_submit_maps_public_contract_without_exposing_run_record():
    manager = _RunManager()
    runtime = EmbeddedRuntime(_container(run_manager=manager))

    record = await runtime.submit(
        RuntimeRunRequest(
            graph_id="agent",
            inputs={"message": "hello"},
            run_id="run-1",
            session_id="session-1",
            tags=("studio:test",),
            identity=RuntimeIdentity(user_id="studio", org_id="tenant-1"),
            origin="playground",
            agent_id="agent-1",
            run_config={"origin_binding": {"source": "studio"}},
        )
    )

    graph_id, submitted = manager.submitted
    assert graph_id == "agent"
    assert submitted["identity"].user_id == "studio"
    assert submitted["identity"].org_id == "tenant-1"
    assert submitted["origin"] is RunOrigin.playground
    assert record.run_id == "run-1"
    assert record.status == "pending"
    assert record.metadata == {"accepted": True}


@pytest.mark.asyncio
async def test_session_artifact_staging_uses_persisted_session_scope(
    monkeypatch,
) -> None:
    canonical_scope = SimpleNamespace(
        as_filter=lambda: {
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "session_id": "session-1",
        }
    )
    observed = {}

    class _Sessions:
        async def storage_scope(self, session_id):
            assert session_id == "session-1"
            return canonical_scope

    class _Stager:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def stage_bytes(self, data, **kwargs):
            observed.update(data=data, stage=kwargs)
            return SimpleNamespace(
                artifact_id="artifact-1",
                size=len(data),
                uri="artifact://artifact-1",
            )

    monkeypatch.setattr(
        "aethergraph.runtime.embedded.ResourceStager",
        _Stager,
    )
    runtime = EmbeddedRuntime(_container(session_store=_Sessions()))

    artifact = await runtime.stage_artifact(
        identity=RuntimeIdentity(user_id="studio-ai:thread-1", org_id="tenant-1"),
        data=b"current buffer",
        name="weather.py",
        mime="text/x-python",
        file_id="buffer-1",
        scope=RuntimeArtifactScope(
            source="agstudio",
            session_id="session-1",
            graph_id="integration",
            node_id="resource_ingress",
            tool_name="integration.resource_ingress",
        ),
    )

    assert observed["identity"] is None
    assert observed["storage_scope"] is canonical_scope
    assert artifact.artifact_id == "artifact-1"


def test_runtime_exposes_immutable_profile_and_capture_values():
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            default=SimpleNamespace(provider="lmstudio", model="agent-engine"),
            profiles={
                "summarizer": SimpleNamespace(
                    provider="openai",
                    model="summary-model",
                )
            },
            observability=SimpleNamespace(capture_mode="manifest"),
        )
    )
    runtime = EmbeddedRuntime(_container(settings=settings))

    default = runtime.model_profile("default")
    summarizer = runtime.model_profile("summarizer")

    assert (default.provider, default.model) == ("lmstudio", "agent-engine")
    assert (summarizer.provider, summarizer.model) == ("openai", "summary-model")
    assert runtime.observability_capture_mode() == "manifest"


@pytest.mark.asyncio
async def test_query_events_merges_exact_run_membership_by_shared_cursor():
    runtime = EmbeddedRuntime(
        _container(
            storage_services=SimpleNamespace(
                runtime_output=_RuntimeOutput(
                    {
                        "root": ({"_row_id": 3, "kind": "root"},),
                        "child": (
                            {"_row_id": 2, "kind": "child-start"},
                            {"_row_id": 4, "kind": "child-end"},
                        ),
                    }
                )
            ),
        )
    )

    rows = await runtime.query_events(run_ids=("root", "child"), limit=2)

    assert [row["_row_id"] for row in rows] == [2, 3]


@pytest.mark.asyncio
async def test_closed_runtime_rejects_new_operations():
    runtime = EmbeddedRuntime(_container())

    await runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        await runtime.query_events(run_ids=("run-1",))
