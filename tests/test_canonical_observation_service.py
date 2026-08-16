from __future__ import annotations

from datetime import UTC, datetime
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.observability.canonical_service import (
    CanonicalObservationService,
    ProviderObservationService,
    bind_canonical_observation_service,
)
from aethergraph.observability.models import (
    LLMObservationRecord,
    ObservationRecord,
    ObservationScope,
)
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.services.llm.provider_transport.models import (
    ProviderRateLimitSnapshot,
    ProviderTransportAttempt,
)
from aethergraph.storage.contracts import (
    ObservationQuery,
    ObservationResourceRelation,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 23, tzinfo=UTC)
OWNER = StorageScope(project_id="project-1", org_id="org-1", user_id="user-1")
RUNTIME_SCOPE = ObservationScope(
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
    app_id="legacy-app",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
    agent_id="agent-1",
    trace_id="trace-1",
    turn_id="turn-1",
)


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _service(
    repository: LocalObservationRepository,
    *,
    capture_mode: str = "manifest",
) -> ProviderObservationService:
    return ProviderObservationService(
        repository=repository,
        owner_scope=OWNER,
        policy=ObservationPolicy(capture_mode=capture_mode),
    )


@pytest.mark.asyncio
async def test_canonical_observation_projection_preserves_producer_links_and_app_envelope(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = LocalObservationRepository(database=database)
    service = _service(repository)
    record = ObservationRecord(
        observation_id="obs-1",
        category="service_operation",
        name="save",
        summary="artifacts/save end",
        occurred_at=NOW.isoformat(),
        scope=RUNTIME_SCOPE,
        attributes={"service": "artifacts", "operation": "save"},
    )

    observation_id = await service.append_observation(
        record,
        resource_links=(
            {
                "resource_key": "artifact:a1",
                "relation": "output",
                "revision": "2",
            },
        ),
    )
    stored = await repository.get(OWNER, observation_id)

    assert stored is not None
    assert stored.producer == "artifacts"
    assert stored.scope.run_id == "run-1"
    assert "app_id" not in stored.scope.as_filter()
    assert stored.attributes["compatibility_metadata"]["app_id"] == {
        "value": "legacy-app",
        "deprecated": True,
        "compatibility_only": True,
    }
    assert stored.resource_links[0].relation is ObservationResourceRelation.OUTPUT
    assert stored.resource_links[0].resource_revision == "2"
    page = await repository.query(
        ObservationQuery(scope=OWNER, producers=("artifacts",), names=("save",))
    )
    assert page.items == (stored,)

    with pytest.raises(ValueError, match="reserve compatibility_metadata"):
        await service.append_observation(
            ObservationRecord(
                observation_id="reserved",
                category="trace",
                name="invalid",
                summary="invalid",
                occurred_at=NOW.isoformat(),
                scope=RUNTIME_SCOPE,
                attributes={"compatibility_metadata": {}},
            )
        )
    await database.close()


@pytest.mark.asyncio
async def test_canonical_llm_projection_is_atomic_idempotent_and_capture_bounded(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = LocalObservationRepository(database=database)
    service = _service(repository)
    call = LLMObservationRecord(
        llm_call_id="call-1",
        created_at=NOW.isoformat(),
        call_type="chat",
        provider="openai",
        model="gpt-test",
        scope=RUNTIME_SCOPE,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort=None,
        max_output_tokens=100,
        output_format="text",
        json_schema=None,
        schema_name=None,
        strict_schema=None,
        validate_json=None,
        extra_params={},
        request_args={"temperature": 0},
        provider_request_args={"temperature": 0},
        compatibility_notes=[],
        trace_payload={"step": "complete"},
        raw_text="hello back",
        usage={"input_tokens": 3, "output_tokens": 2},
        latency_ms=25,
        attempts=(
            ProviderTransportAttempt(
                attempt_number=1,
                elapsed_s=0.025,
                outcome="success",
                retryable=False,
                status_code=200,
                rate_limits=(ProviderRateLimitSnapshot(resource="requests", remaining=99),),
            ),
        ),
    )

    await service.emit(call, capture_mode="manifest")
    await service.emit(call, capture_mode="manifest")
    detail = await repository.get_llm_call(OWNER, "call-1")

    assert detail is not None
    assert detail.record.observation.producer == "aethergraph.llm"
    assert (
        detail.record.observation.attributes["compatibility_metadata"]["app_id"]["value"]
        == "legacy-app"
    )
    assert detail.record.prompt_manifest_id == "llm-manifest:call-1"
    assert detail.captured_request["messages"][0]["content"] == "hello"
    assert detail.captured_response == {"text": "hello back"}
    assert detail.trace_payload == {"step": "complete"}
    assert detail.record.attempts[0].elapsed_ms == 25
    assert call.prompt_manifest_id == detail.record.prompt_manifest_id
    with pytest.raises(ValueError, match="does not match"):
        await service.emit(call, capture_mode="full")
    await database.close()


def test_canonical_observation_binding_is_exact_inactive_and_strictly_documented() -> None:
    repository = object()
    bundle = SimpleNamespace(observations=repository)
    service = bind_canonical_observation_service(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=OWNER,
        policy=ObservationPolicy(capture_mode="off"),
    )

    assert isinstance(service, ProviderObservationService)
    assert service.repository is repository
    assert service.owner_scope is OWNER
    with pytest.raises(ValueError, match="execution/external"):
        ProviderObservationService(
            repository=repository,  # type: ignore[arg-type]
            owner_scope=StorageScope(project_id="project-1", run_id="run-1"),
            policy=ObservationPolicy(),
        )

    required = ("Examples:", "Args:", "Returns:", "Notes:")
    methods = (
        *(
            member
            for name, member in inspect.getmembers(CanonicalObservationService, inspect.isfunction)
            if not name.startswith("_")
        ),
        *(
            member
            for name, member in inspect.getmembers(ProviderObservationService, inspect.isfunction)
            if not name.startswith("_")
        ),
        bind_canonical_observation_service,
    )
    for method in methods:
        docstring = inspect.getdoc(method) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), method.__name__
        assert positions == tuple(sorted(positions)), method.__name__
        assert docstring.count("```python") >= 2, method.__name__
