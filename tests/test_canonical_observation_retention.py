from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.observability.canonical_retention import (
    ProviderRetentionJanitor,
    RetentionPolicy,
)
from aethergraph.observability.canonical_service import ProviderObservationService
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.storage.contracts import (
    LLMCallDraft,
    LLMCallLifecycleStatus,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationQuery,
    ObservationScopeManagementRecord,
    ObservationSeverity,
    ObservationStatus,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 12, tzinfo=UTC)
OWNER = StorageScope(project_id="project-1")


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _draft(
    observation_id: str,
    *,
    occurred_at: datetime,
    trace_id: str,
    run_id: str,
    severity: ObservationSeverity = ObservationSeverity.INFO,
    expires_at: datetime | None = None,
    category: str = "trace",
) -> ObservationDraft:
    return ObservationDraft(
        observation_id=observation_id,
        category=category,
        name=observation_id,
        summary=observation_id,
        occurred_at=occurred_at,
        scope=StorageScope(project_id="project-1", run_id=run_id),
        status=(
            ObservationStatus.ERROR
            if severity in {ObservationSeverity.ERROR, ObservationSeverity.CRITICAL}
            else ObservationStatus.OK
        ),
        severity=severity,
        trace_id=trace_id,
        expires_at=expires_at,
    )


def _service(repository: LocalObservationRepository) -> ProviderObservationService:
    return ProviderObservationService(
        repository=repository,
        owner_scope=OWNER,
        policy=ObservationPolicy(),
    )


@pytest.mark.asyncio
async def test_provider_retention_applies_age_expiry_error_and_pin_rules(tmp_path: Path) -> None:
    database = _database(tmp_path)
    repository = LocalObservationRepository(database=database)
    service = _service(repository)
    full_observation = _draft(
        "obs-full",
        occurred_at=NOW - timedelta(days=4),
        trace_id="trace-full",
        run_id="run-full",
        category="llm",
    )
    completed_call = LLMCallDraft(
        llm_call_id="call-full",
        observation=replace(full_observation, retention_class="forensic"),
        call_type="chat",
        provider="openai",
        model="model",
        capture_mode=ObservationCaptureMode.FULL,
        prompt_manifest_id="manifest-full",
        captured_request={"messages": []},
    )
    started_call = replace(
        completed_call,
        observation=replace(
            completed_call.observation,
            status=ObservationStatus.PENDING,
            severity=ObservationSeverity.INFO,
        ),
        lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
    )
    await repository.begin_llm_call(started_call)
    await repository.finish_llm_call("call-full", completed_call)
    await repository.append_many(
        (
            _draft(
                "obs-expired",
                occurred_at=NOW - timedelta(days=2),
                trace_id="trace-expired",
                run_id="run-expired",
                expires_at=NOW - timedelta(days=1),
            ),
            _draft(
                "obs-normal-old",
                occurred_at=NOW - timedelta(days=31),
                trace_id="trace-normal",
                run_id="run-normal",
            ),
            _draft(
                "obs-error-young",
                occurred_at=NOW - timedelta(days=31),
                trace_id="trace-error-young",
                run_id="run-error-young",
                severity=ObservationSeverity.ERROR,
            ),
            _draft(
                "obs-error-old",
                occurred_at=NOW - timedelta(days=91),
                trace_id="trace-error-old",
                run_id="run-error-old",
                severity=ObservationSeverity.CRITICAL,
            ),
            _draft(
                "obs-pinned",
                occurred_at=NOW - timedelta(days=40),
                trace_id="trace-pinned",
                run_id="run-pinned",
            ),
        )
    )
    await repository.compare_and_set_scope_management(
        ObservationScopeManagementRecord(
            scope_key="trace:trace-pinned",
            scope=OWNER,
            revision=1,
            updated_at=NOW,
            trace_id="trace-pinned",
            pinned=True,
        ),
        0,
    )
    janitor = ProviderRetentionJanitor(
        service,
        RetentionPolicy(
            max_bytes_per_trace=10**9,
            max_total_bytes=10**9,
        ),
    )

    results = await janitor.run_once(now=NOW)
    remaining = await repository.query(ObservationQuery(scope=OWNER))

    assert [result.deleted_observations for result in results[:4]] == [1, 1, 1, 1]
    assert all(not result.dry_run for result in results)
    assert {item.observation_id for item in remaining.items} == {
        "obs-error-young",
        "obs-pinned",
    }
    await database.close()


@pytest.mark.asyncio
async def test_provider_retention_pages_scope_counts_and_preserves_nested_pin(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    repository = LocalObservationRepository(database=database)
    service = _service(repository)
    await repository.append_many(
        tuple(
            _draft(
                f"obs-{index}",
                occurred_at=NOW + timedelta(seconds=index),
                trace_id=f"trace-{index}",
                run_id=f"run-{index}",
            )
            for index in range(1, 4)
        )
    )
    await repository.compare_and_set_scope_management(
        ObservationScopeManagementRecord(
            scope_key="trace:trace-1",
            scope=StorageScope(project_id="project-1", run_id="run-1"),
            revision=1,
            updated_at=NOW,
            trace_id="trace-1",
            pinned=True,
        ),
        0,
    )
    janitor = ProviderRetentionJanitor(
        service,
        RetentionPolicy(
            max_age_days=365,
            error_max_age_days=365,
            max_full_prompt_age_days=365,
            max_bytes_per_trace=10**9,
            max_total_bytes=10**9,
            max_retained_traces=1,
            max_retained_runs=1,
        ),
        scope_action_limit=10,
    )

    await janitor.run_once(now=NOW + timedelta(hours=1))
    remaining = await repository.query(ObservationQuery(scope=OWNER))

    assert {item.observation_id for item in remaining.items} == {"obs-3", "obs-1"}
    stop_event = asyncio.Event()
    stop_event.set()
    await janitor.run_forever(stop_event)
    await database.close()


def test_provider_retention_public_docstrings_and_bounds_are_strict() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    methods = (
        ProviderRetentionJanitor.__init__,
        ProviderRetentionJanitor.run_once,
        ProviderRetentionJanitor.run_forever,
    )
    for method in methods:
        docstring = inspect.getdoc(method) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), method.__name__
        assert positions == tuple(sorted(positions)), method.__name__
        assert docstring.count("```python") >= 2, method.__name__

    service = object()
    with pytest.raises(ValueError, match="scope_action_limit"):
        ProviderRetentionJanitor(
            service,  # type: ignore[arg-type]
            RetentionPolicy(),
            scope_action_limit=0,
        )
    with pytest.raises(ValueError, match="error_max_age_days"):
        ProviderRetentionJanitor(
            service,  # type: ignore[arg-type]
            RetentionPolicy(max_age_days=90, error_max_age_days=30),
        )
