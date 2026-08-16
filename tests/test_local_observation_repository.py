from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest
from storage_conformance.suite import (
    NOW as CONFORMANCE_NOW,
    check_observation_summary_conformance,
)

from aethergraph.storage.contracts import (
    LLMCallAttempt,
    LLMCallDraft,
    LLMCallQuery,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationLLMSummaryQuery,
    ObservationPurgeRequest,
    ObservationQuery,
    ObservationResourceLink,
    ObservationResourceRelation,
    ObservationScopeManagementQuery,
    ObservationScopeManagementRecord,
    ObservationScopeUsageQuery,
    ObservationSeverity,
    ObservationStatus,
    ObservationTraceSummaryQuery,
    ObservationUsageDimension,
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 22, tzinfo=UTC)
SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
    agent_id="agent-1",
    scope_key="scope-1",
)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _observation(
    observation_id: str,
    *,
    scope: StorageScope = SCOPE,
    category: str = "trace",
    occurred_at: datetime = NOW,
    trace_id: str | None = "trace-1",
    status: ObservationStatus = ObservationStatus.OK,
    severity: ObservationSeverity = ObservationSeverity.INFO,
    producer: str | None = "aethergraph.runtime",
    attributes: dict | None = None,
    retention_class: str = "standard",
    resource_links: tuple[ObservationResourceLink, ...] = (),
) -> ObservationDraft:
    return ObservationDraft(
        observation_id=observation_id,
        category=category,
        name=f"name-{observation_id}",
        summary=f"summary-{observation_id}",
        occurred_at=occurred_at,
        scope=scope,
        trace_id=trace_id,
        turn_id="turn-1",
        status=status,
        severity=severity,
        producer=producer,
        attributes=(
            {"index": observation_id, "nested": {"ok": True}} if attributes is None else attributes
        ),
        resource_links=resource_links,
        retention_class=retention_class,
    )


def _llm_call(
    llm_call_id: str,
    *,
    scope: StorageScope = SCOPE,
    observation_id: str | None = None,
    manifest_id: str | None = None,
    occurred_at: datetime = NOW,
    trace_id: str = "trace-1",
    capture_mode: ObservationCaptureMode = ObservationCaptureMode.FULL,
    provider: str = "openai",
    model: str = "gpt-test",
    usage: dict | None = None,
    error_type: str | None = None,
    captured_request: object = None,
    captured_response: object = None,
    trace_payload: object = None,
) -> LLMCallDraft:
    if observation_id is None:
        observation_id = f"obs-{llm_call_id}"
    if capture_mode is not ObservationCaptureMode.OFF and manifest_id is None:
        manifest_id = f"manifest-{llm_call_id}"
    if capture_mode in {ObservationCaptureMode.OFF, ObservationCaptureMode.METADATA}:
        captured_request = captured_response = trace_payload = None
    return LLMCallDraft(
        llm_call_id=llm_call_id,
        observation=_observation(
            observation_id,
            scope=scope,
            category="llm",
            occurred_at=occurred_at,
            trace_id=trace_id,
            status=(ObservationStatus.ERROR if error_type is not None else ObservationStatus.OK),
        ),
        call_type="chat",
        provider=provider,
        model=model,
        capture_mode=capture_mode,
        profile_name="default",
        call_name="answer",
        request_options={"temperature": 0},
        usage={"input_tokens": 5, "output_tokens": 3} if usage is None else usage,
        latency_ms=25,
        error_type=error_type,
        error_message="failed" if error_type is not None else None,
        prompt_manifest_id=manifest_id,
        request_preview={"messages": 1},
        response_preview={"chars": 2},
        captured_request=captured_request,
        captured_response=captured_response,
        trace_payload=trace_payload,
        attempts=(
            LLMCallAttempt(
                attempt_number=1,
                elapsed_ms=25,
                outcome="success",
                retryable=False,
                status_code=200,
                request_id=f"request-{llm_call_id}",
                rate_limits=({"remaining": 99},),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_provider_side_trace_summary_is_scoped_bounded_and_legacy_ordered(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    other_scope = replace(SCOPE, run_id="other-run", scope_key="other-scope")
    await repository.append_many(
        (
            _observation(
                "outside-window",
                occurred_at=NOW - timedelta(seconds=1),
                attributes={"duration_ms": 100, "error": {"type": "Old"}},
            ),
            _observation(
                "trace-z",
                occurred_at=NOW,
                trace_id="trace-z",
                producer="runner",
                status=ObservationStatus.ERROR,
                attributes={"duration_ms": 7, "error": {"type": "RuntimeError"}},
            ),
            _observation(
                "trace-a",
                occurred_at=NOW + timedelta(seconds=1),
                trace_id="trace-a",
                producer=None,
                status=ObservationStatus.ERROR,
                attributes={
                    "duration_ms": 11,
                    "service": "runner",
                    "error": {"type": "ValueError"},
                },
            ),
            _observation(
                "run-fallback",
                occurred_at=NOW + timedelta(seconds=2),
                trace_id=None,
                attributes={"duration_ms": 3},
            ),
            _observation(
                "ignored-log",
                category="log",
                occurred_at=NOW + timedelta(seconds=2),
                attributes={"duration_ms": 200, "error": {"type": "Ignored"}},
            ),
            _observation(
                "other-run",
                scope=other_scope,
                occurred_at=NOW + timedelta(seconds=2),
                attributes={"duration_ms": 300, "error": {"type": "Ignored"}},
            ),
        )
    )

    summary = await repository.summarize_traces(
        ObservationTraceSummaryQuery(
            scope=StorageScope(project_id="project-1", run_id="run-1"),
            occurred_at_or_after=NOW,
            occurred_at_or_before=NOW + timedelta(seconds=2),
            trace_id_limit=2,
            failing_service_limit=1,
        )
    )

    assert summary.span_count == 3
    assert summary.error_count == 2
    assert summary.total_duration_ms == 21
    assert summary.trace_id_count == 3
    assert summary.trace_ids == ("run-1", "trace-a")
    assert summary.trace_ids_truncated
    assert dict(summary.top_failing_services) == {"runner": 2}
    assert summary.latest_error_at == NOW + timedelta(seconds=1)
    await database.close()


@pytest.mark.asyncio
async def test_provider_side_llm_summary_preserves_token_and_model_semantics(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    await repository.append_llm_call(
        _llm_call(
            "outside",
            occurred_at=NOW - timedelta(seconds=1),
            model="outside",
            usage={"total_tokens": 1000},
        )
    )
    await repository.append_llm_call(
        _llm_call(
            "alias",
            occurred_at=NOW,
            model="model-z",
            usage={"input_tokens": 5, "output_tokens": 3},
        )
    )
    await repository.append_llm_call(
        _llm_call(
            "direct",
            occurred_at=NOW + timedelta(seconds=1),
            model="model-a",
            usage={"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 20},
            error_type="ProviderError",
        )
    )
    await repository.append_llm_call(
        _llm_call(
            "newest",
            occurred_at=NOW + timedelta(seconds=2),
            model="model-z",
            usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 0},
        )
    )

    query = ObservationLLMSummaryQuery(
        scope=StorageScope(project_id="project-1", run_id="run-1"),
        occurred_at_or_after=NOW,
        model_limit=1,
    )
    summary = await repository.summarize_llm_calls(query)
    fallback = await repository.summarize_llm_calls(
        replace(query, occurred_at_or_after=NOW + timedelta(seconds=2))
    )

    assert summary.total_calls == 3
    assert summary.total_prompt_tokens == 14
    assert summary.total_completion_tokens == 8
    assert summary.total_tokens == 20
    assert summary.error_count == 1
    assert summary.model_count == 2
    assert dict(summary.by_model) == {"model-z": 2}
    assert summary.by_model_truncated
    assert fallback.total_tokens == 3
    await database.close()


@pytest.mark.asyncio
async def test_trace_summary_high_cardinality_returns_exact_count_and_bounded_ids(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    await repository.append_many(
        tuple(
            _observation(
                f"cardinality-{index:03d}",
                trace_id=f"trace-{index:03d}",
                occurred_at=NOW + timedelta(milliseconds=index),
                attributes={"duration_ms": 1},
            )
            for index in range(125)
        )
    )

    summary = await repository.summarize_traces(
        ObservationTraceSummaryQuery(scope=StorageScope(project_id="project-1", run_id="run-1"))
    )

    assert summary.span_count == 125
    assert summary.trace_id_count == 125
    assert len(summary.trace_ids) == 100
    assert summary.trace_ids[:2] == ("trace-000", "trace-001")
    assert summary.trace_ids[-1] == "trace-099"
    assert summary.trace_ids_truncated
    await database.close()


@pytest.mark.asyncio
async def test_local_observation_summaries_pass_shared_provider_conformance(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    scope = replace(SCOPE, run_id="summary-run-1", scope_key="summary-scope")
    await repository.append_many(
        (
            _observation(
                "conformance-trace-a",
                scope=scope,
                occurred_at=CONFORMANCE_NOW,
                trace_id="trace-a",
                producer="runner",
                status=ObservationStatus.ERROR,
                attributes={"duration_ms": 7, "error": {"type": "RuntimeError"}},
            ),
            _observation(
                "conformance-trace-b",
                scope=scope,
                occurred_at=CONFORMANCE_NOW,
                trace_id="trace-b",
                attributes={"duration_ms": 5},
            ),
        )
    )
    await repository.append_llm_call(
        _llm_call(
            "conformance-a",
            scope=scope,
            occurred_at=CONFORMANCE_NOW,
            model="model-a",
            usage={"input_tokens": 2, "output_tokens": 1},
        )
    )
    await repository.append_llm_call(
        _llm_call(
            "conformance-b",
            scope=scope,
            occurred_at=CONFORMANCE_NOW,
            model="model-b",
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            error_type="ProviderError",
        )
    )

    await check_observation_summary_conformance(repository)
    await database.close()


@pytest.mark.asyncio
async def test_observation_append_order_idempotency_scope_and_atomic_conflict(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    drafts = tuple(
        _observation(f"obs-{index}", occurred_at=NOW + timedelta(seconds=index))
        for index in range(3)
    )

    stored = await repository.append_many(drafts)
    assert [item.observation_id for item in stored] == [item.observation_id for item in drafts]
    assert len({item.cursor for item in stored}) == 3
    assert await repository.append_many(drafts) == stored
    assert await repository.get(StorageScope(project_id="other"), "obs-0") is None
    assert await repository.get(StorageScope(), "obs-0") is None
    with pytest.raises(StorageConfigurationError, match="non-empty tuple"):
        await repository.append_many(())

    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.append_many(
            (
                _observation("rolled-back"),
                replace(drafts[0], summary="conflicting content"),
            )
        )
    assert await repository.get(SCOPE, "rolled-back") is None
    await database.close()


@pytest.mark.asyncio
async def test_concurrent_observation_appends_assign_unique_cursors(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    results = await asyncio.gather(
        *(repository.append_many((_observation(f"obs-{index}"),)) for index in range(20))
    )
    records = tuple(batch[0] for batch in results)
    assert len({record.cursor for record in records}) == 20
    assert len((await repository.query(ObservationQuery(scope=SCOPE))).items) == 20
    await database.close()


@pytest.mark.asyncio
async def test_observation_query_filters_resource_without_duplicates_and_binds_cursor(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    links = (
        ObservationResourceLink(
            resource_key="artifact:1", relation=ObservationResourceRelation.INPUT
        ),
        ObservationResourceLink(
            resource_key="artifact:1", relation=ObservationResourceRelation.OUTPUT
        ),
    )
    drafts = (
        _observation("obs-old", occurred_at=NOW, resource_links=links),
        _observation(
            "obs-warning",
            occurred_at=NOW + timedelta(seconds=1),
            status=ObservationStatus.PENDING,
            severity=ObservationSeverity.WARNING,
        ),
        _observation("obs-new", occurred_at=NOW + timedelta(seconds=2)),
    )
    await repository.append_many(drafts)

    resource_page = await repository.query(ObservationQuery(scope=SCOPE, resource_key="artifact:1"))
    assert [item.observation_id for item in resource_page.items] == ["obs-old"]
    producer_page = await repository.query(
        ObservationQuery(
            scope=SCOPE,
            names=("name-obs-new",),
            producers=("aethergraph.runtime",),
        )
    )
    assert [item.observation_id for item in producer_page.items] == ["obs-new"]
    relation_page = await repository.query(
        ObservationQuery(
            scope=SCOPE,
            resource_key="artifact:1",
            resource_relation=ObservationResourceRelation.OUTPUT,
        )
    )
    assert relation_page.items[0].resource_links == links
    filtered = await repository.query(
        ObservationQuery(
            scope=SCOPE,
            statuses=(ObservationStatus.PENDING,),
            severities=(ObservationSeverity.WARNING,),
            occurred_at_or_after=NOW + timedelta(seconds=1),
            occurred_at_or_before=NOW + timedelta(seconds=1),
        )
    )
    assert [item.observation_id for item in filtered.items] == ["obs-warning"]

    query = ObservationQuery(scope=SCOPE, page=PageRequest(limit=2))
    first = await repository.query(query)
    second = await repository.query(
        replace(query, page=PageRequest(limit=2, cursor=first.next_cursor))
    )
    assert [item.observation_id for item in (*first.items, *second.items)] == [
        "obs-new",
        "obs-warning",
        "obs-old",
    ]
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query(
            replace(
                query,
                trace_id="different",
                page=PageRequest(limit=2, cursor=first.next_cursor),
            )
        )
    await database.close()


@pytest.mark.asyncio
async def test_llm_full_capture_is_atomic_idempotent_and_detail_only(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    call = _llm_call(
        "call-1",
        captured_request={"messages": [{"role": "user", "content": "hi"}]},
        captured_response={"text": "ok"},
        trace_payload={"steps": ["request", "response"]},
    )

    record = await repository.append_llm_call(call)
    assert record.llm_call_id == "call-1"
    assert record.attempts == call.attempts
    assert not hasattr(record, "captured_request")
    assert await repository.append_llm_call(call) == record
    page = await repository.query_llm_calls(LLMCallQuery(scope=SCOPE))
    assert page.items == (record,)
    assert not hasattr(page.items[0], "captured_response")
    detail = await repository.get_llm_call(SCOPE, "call-1")
    assert detail is not None
    assert detail.captured_request == call.captured_request
    assert detail.captured_response == call.captured_response
    assert detail.trace_payload == call.trace_payload
    assert await repository.get_llm_call(StorageScope(run_id="other"), "call-1") is None

    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.append_llm_call(replace(call, model="different"))
    await repository.append_many((_observation("preexisting", category="llm"),))
    with pytest.raises(StorageIntegrityError, match="not created atomically"):
        await repository.append_llm_call(
            _llm_call("call-2", observation_id="preexisting", captured_response="no")
        )
    assert await repository.get_llm_call(SCOPE, "call-2") is None
    await database.close()


@pytest.mark.asyncio
async def test_off_and_metadata_capture_never_retain_or_hydrate_content(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    calls = (
        _llm_call("off", capture_mode=ObservationCaptureMode.OFF, manifest_id=None),
        _llm_call("metadata", capture_mode=ObservationCaptureMode.METADATA),
    )
    for call in calls:
        await repository.append_llm_call(call)
        detail = await repository.get_llm_call(SCOPE, call.llm_call_id)
        assert detail is not None
        assert detail.captured_request is None
        assert detail.captured_response is None
        assert detail.trace_payload is None
    stats = await repository.storage_stats(SCOPE)
    assert stats.observations == stats.llm_calls == 2
    assert stats.manifests == 1
    assert stats.fragments == stats.fragment_bytes == 0
    await database.close()


@pytest.mark.asyncio
async def test_capture_fragments_deduplicate_and_purge_preserves_shared_content(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    content = {
        "captured_request": {"messages": ["same"]},
        "captured_response": {"text": "same"},
        "trace_payload": {"trace": "same"},
    }
    first = _llm_call("call-1", occurred_at=NOW, trace_id="trace-1", **content)
    second = _llm_call(
        "call-2",
        occurred_at=NOW + timedelta(seconds=1),
        trace_id="trace-2",
        **content,
    )
    await repository.append_llm_call(first)
    await repository.append_llm_call(second)
    stats = await repository.storage_stats(SCOPE)
    assert stats.manifests == 2
    assert stats.fragments == 3

    request = ObservationPurgeRequest(
        scope=SCOPE,
        trace_id="trace-1",
        occurred_before=NOW + timedelta(seconds=1),
    )
    preview = await repository.purge(request)
    assert preview.dry_run
    assert preview.matching_observations == preview.matching_manifests == 1
    assert preview.shared_fragment_bytes_retained == stats.fragment_bytes
    assert preview.exclusive_fragment_bytes == 0
    executed = await repository.purge(replace(request, dry_run=False))
    assert executed.deleted_observations == executed.deleted_manifests == 1
    assert executed.deleted_fragments == 0
    assert await repository.get_llm_call(SCOPE, "call-1") is None
    remaining = await repository.get_llm_call(SCOPE, "call-2")
    assert remaining is not None and remaining.captured_response == content["captured_response"]

    final = await repository.purge(
        ObservationPurgeRequest(scope=SCOPE, trace_id="trace-2", dry_run=False)
    )
    assert final.deleted_observations == final.deleted_manifests == 1
    assert final.deleted_fragments == 3
    assert (await repository.storage_stats(SCOPE)).fragments == 0
    await database.close()


@pytest.mark.asyncio
async def test_purge_is_bounded_target_aware_and_excludes_pinned_trace(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    await repository.append_many(
        tuple(
            _observation(
                f"obs-{index}",
                occurred_at=NOW + timedelta(seconds=index),
                trace_id="trace-pinned" if index == 0 else f"trace-{index}",
            )
            for index in range(5)
        )
    )
    pinned = ObservationScopeManagementRecord(
        scope_key="trace:trace-pinned",
        scope=StorageScope(project_id=SCOPE.project_id),
        revision=1,
        updated_at=NOW,
        trace_id="trace-pinned",
        pinned=True,
    )
    await repository.compare_and_set_scope_management(pinned, 0)

    bounded = await repository.purge(ObservationPurgeRequest(scope=SCOPE, max_observations=2))
    assert bounded.matching_observations == 2
    assert bounded.matching_traces == 2
    assert await repository.get(SCOPE, "obs-0") is not None
    target = await repository.purge(ObservationPurgeRequest(scope=SCOPE, target_reclaimed_bytes=1))
    assert target.matching_observations == 1
    executed = await repository.purge(
        ObservationPurgeRequest(scope=SCOPE, max_observations=10, dry_run=False)
    )
    assert executed.deleted_observations == 4
    assert await repository.get(SCOPE, "obs-0") is not None
    no_op = await repository.purge(
        ObservationPurgeRequest(
            scope=SCOPE,
            trace_id="missing",
            dry_run=False,
        )
    )
    assert not no_op.dry_run
    assert no_op.deleted_observations == 0
    await database.close()


@pytest.mark.asyncio
async def test_purge_filters_capture_retention_and_severity_in_provider(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    full = _llm_call("full", capture_mode=ObservationCaptureMode.FULL)
    metadata = _llm_call("metadata", capture_mode=ObservationCaptureMode.METADATA)
    await repository.append_llm_call(full)
    await repository.append_llm_call(metadata)
    await repository.append_many(
        (
            _observation(
                "forensic-info",
                retention_class="forensic",
                severity=ObservationSeverity.INFO,
            ),
            _observation(
                "forensic-error",
                retention_class="forensic",
                severity=ObservationSeverity.ERROR,
            ),
        )
    )

    capture = await repository.purge(
        ObservationPurgeRequest(
            scope=SCOPE,
            capture_modes=(ObservationCaptureMode.FULL,),
        )
    )
    assert capture.matching_observations == 1
    retained = await repository.purge(
        ObservationPurgeRequest(
            scope=SCOPE,
            retention_classes=("forensic",),
            excluded_severities=(ObservationSeverity.ERROR,),
        )
    )
    assert retained.matching_observations == 1
    await database.close()


@pytest.mark.asyncio
async def test_scope_usage_and_management_queries_are_bounded_and_cursor_bound(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    project_scope = StorageScope(project_id=SCOPE.project_id)
    second_scope = replace(SCOPE, run_id="run-2", session_id="session-2")
    shared = {
        "captured_request": {"messages": ["shared"]},
        "captured_response": {"text": "shared"},
    }
    await repository.append_llm_call(
        _llm_call("usage-1", trace_id="trace-1", scope=SCOPE, **shared)
    )
    await repository.append_llm_call(
        _llm_call(
            "usage-2",
            trace_id="trace-2",
            scope=second_scope,
            occurred_at=NOW + timedelta(seconds=1),
            **shared,
        )
    )
    pinned = ObservationScopeManagementRecord(
        scope_key="trace:trace-1",
        scope=project_scope,
        revision=1,
        updated_at=NOW,
        trace_id="trace-1",
        pinned=True,
        hidden=True,
    )
    deleted = ObservationScopeManagementRecord(
        scope_key="run:run-2",
        scope=second_scope,
        revision=1,
        updated_at=NOW + timedelta(seconds=1),
        deleted=True,
    )
    await repository.compare_and_set_scope_management(pinned, 0)
    await repository.compare_and_set_scope_management(deleted, 0)

    trace_query = ObservationScopeUsageQuery(
        scope=project_scope,
        dimension=ObservationUsageDimension.TRACE,
        page=PageRequest(limit=1),
    )
    first = await repository.query_scope_usage(trace_query)
    second = await repository.query_scope_usage(
        replace(trace_query, page=PageRequest(limit=1, cursor=first.next_cursor))
    )
    usage = (*first.items, *second.items)
    assert [item.scope_id for item in usage] == ["trace-2", "trace-1"]
    assert all(item.logical_bytes > 0 and item.observation_count == 1 for item in usage)
    assert usage[1].pinned
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query_scope_usage(
            replace(
                trace_query,
                dimension=ObservationUsageDimension.RUN,
                page=PageRequest(limit=1, cursor=first.next_cursor),
            )
        )

    suppressed = await repository.query_scope_management(
        ObservationScopeManagementQuery(scope=project_scope, hidden=True)
    )
    assert suppressed.items == (pinned,)
    deleted_page = await repository.query_scope_management(
        ObservationScopeManagementQuery(scope=project_scope, deleted=True)
    )
    assert deleted_page.items == (deleted,)
    await database.close()


@pytest.mark.asyncio
async def test_scope_management_cas_exact_identity_and_immutability(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    record = ObservationScopeManagementRecord(
        scope_key="trace:1",
        scope=SCOPE,
        revision=1,
        updated_at=NOW,
        trace_id="trace-1",
        pinned=True,
        tags=("important",),
    )
    assert await repository.compare_and_set_scope_management(record, 0) == record
    assert await repository.get_scope_management(SCOPE, record.scope_key) == record
    assert (
        await repository.get_scope_management(
            StorageScope(project_id=SCOPE.project_id), record.scope_key
        )
        is None
    )
    updated = replace(
        record,
        revision=2,
        updated_at=NOW + timedelta(seconds=1),
        hidden=True,
    )
    assert await repository.compare_and_set_scope_management(updated, 1) == updated
    with pytest.raises(StorageConflictError, match="stale"):
        await repository.compare_and_set_scope_management(updated, 1)
    with pytest.raises(StorageIntegrityError, match="immutable"):
        await repository.compare_and_set_scope_management(
            replace(
                updated,
                revision=3,
                updated_at=NOW + timedelta(seconds=2),
                trace_id="different",
            ),
            2,
        )
    await database.close()


@pytest.mark.asyncio
async def test_read_only_observation_repository_allows_reads_and_dry_run_only(
    tmp_path: Path,
) -> None:
    writer_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writer = LocalObservationRepository(database=writer_database)
    call = _llm_call("call-1", captured_request={"input": "x"})
    await writer.append_llm_call(call)
    await writer_database.close()

    database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    repository = LocalObservationRepository(database=database)
    assert await repository.get(SCOPE, call.observation.observation_id) is not None
    assert (await repository.query(ObservationQuery(scope=SCOPE))).items
    assert await repository.get_llm_call(SCOPE, call.llm_call_id) is not None
    assert (await repository.query_llm_calls(LLMCallQuery(scope=SCOPE))).items
    assert (
        await repository.summarize_traces(ObservationTraceSummaryQuery(scope=SCOPE))
    ).span_count == 0
    assert (
        await repository.summarize_llm_calls(ObservationLLMSummaryQuery(scope=SCOPE))
    ).total_calls == 1
    assert (await repository.storage_stats(SCOPE)).llm_calls == 1
    assert (
        await repository.query_scope_usage(
            ObservationScopeUsageQuery(
                scope=SCOPE,
                dimension=ObservationUsageDimension.TRACE,
            )
        )
    ).items
    assert not (
        await repository.query_scope_management(ObservationScopeManagementQuery(scope=SCOPE))
    ).items
    assert (await repository.purge(ObservationPurgeRequest(scope=SCOPE))).dry_run
    with pytest.raises(StorageReadOnlyError):
        await repository.append_many((_observation("new"),))
    with pytest.raises(StorageReadOnlyError):
        await repository.append_llm_call(_llm_call("new"))
    with pytest.raises(StorageReadOnlyError):
        await repository.purge(ObservationPurgeRequest(scope=SCOPE, dry_run=False))
    with pytest.raises(StorageReadOnlyError):
        await repository.compare_and_set_scope_management(
            ObservationScopeManagementRecord(
                scope_key="trace:new", scope=SCOPE, revision=1, updated_at=NOW
            ),
            0,
        )
    await database.close()


@pytest.mark.asyncio
async def test_observation_schema_has_no_legacy_identity_and_promoted_query_indexes(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalObservationRepository(database=database)
    tables = await database.fetch_all(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'local_%'"
    )
    forbidden = {"app_id", "application_id", "client_id", "path"}
    for table in tables:
        columns = await database.fetch_all(f"PRAGMA table_info({table['name']})")
        assert forbidden.isdisjoint({str(column["name"]) for column in columns})

    plans = {
        "run": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observations "
            "WHERE run_id = ? ORDER BY occurred_at DESC, sequence DESC LIMIT ?",
            ("run-1", 10),
        ),
        "resource": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observation_resource_links "
            "WHERE resource_key = ? AND relation = ?",
            ("artifact:1", "input"),
        ),
        "provider": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_llm_calls WHERE provider = ?",
            ("openai",),
        ),
        "producer": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observations "
            "WHERE producer = ? ORDER BY occurred_at DESC, sequence DESC LIMIT ?",
            ("aethergraph.runtime", 10),
        ),
        "name": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observations "
            "WHERE name = ? ORDER BY occurred_at DESC, sequence DESC LIMIT ?",
            ("runner.execute", 10),
        ),
        "capture": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_llm_calls WHERE capture_mode = ?",
            ("full",),
        ),
        "management": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observation_scope_management "
            "WHERE trace_id = ? AND pinned = 1",
            ("trace-1",),
        ),
        "usage": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT trace_id, MAX(occurred_at), MAX(sequence) "
            "FROM local_observations WHERE project_id = ? AND trace_id IS NOT NULL "
            "GROUP BY trace_id",
            ("project-1",),
        ),
        "management_visibility": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observation_scope_management "
            "WHERE project_id = ? AND hidden = ? AND deleted = ? "
            "ORDER BY updated_at DESC, sequence DESC LIMIT ?",
            ("project-1", 1, 0, 10),
        ),
        "trace_summary": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM local_observations o "
            "WHERE o.project_id = ? AND o.run_id = ? "
            "AND o.category IN ('service_operation', 'trace') "
            "AND o.occurred_at >= ?",
            ("project-1", "run-1", NOW.isoformat()),
        ),
        "llm_summary": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM local_llm_calls l "
            "JOIN local_observations o ON o.observation_id = l.observation_id "
            "WHERE o.project_id = ? AND o.run_id = ? AND o.occurred_at >= ?",
            ("project-1", "run-1", NOW.isoformat()),
        ),
        "error_logs": await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_observations o "
            "WHERE o.project_id = ? AND o.category = 'log' "
            "AND o.severity IN ('warning', 'error', 'critical') "
            "ORDER BY o.occurred_at DESC, o.sequence DESC LIMIT ?",
            ("project-1", 100),
        ),
    }
    details = {name: " ".join(str(row[3]) for row in rows) for name, rows in plans.items()}
    assert "ix_local_observations_run_time" in details["run"]
    assert "ix_local_observation_resources_lookup" in details["resource"]
    assert "ix_local_llm_calls_provider" in details["provider"]
    assert "ix_local_observations_producer_time" in details["producer"]
    assert "ix_local_observations_name_time" in details["name"]
    assert "ix_local_llm_calls_capture" in details["capture"]
    assert "ix_local_observation_management_trace" in details["management"]
    assert "ix_local_observations_project_trace_time" in details["usage"]
    assert "ix_local_observation_management_visibility" in details["management_visibility"]
    assert "ix_local_observations_project_run_time" in details["trace_summary"]
    assert "ix_local_observations_project_run_time" in details["llm_summary"]
    assert "ix_local_observations_category_time" in details["error_logs"]
    await database.close()


@pytest.mark.asyncio
async def test_persisted_observation_and_capture_corruption_raise_typed_error(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalObservationRepository(database=database)
    await repository.append_many((_observation("obs-corrupt"),))
    await database.transaction(
        lambda connection: connection.execute(
            "UPDATE local_observations SET attributes_json = '[]' "
            "WHERE observation_id = 'obs-corrupt'"
        )
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await repository.get(SCOPE, "obs-corrupt")

    call = _llm_call("call-corrupt", captured_response={"secret": "value"})
    await repository.append_llm_call(call)
    await database.transaction(
        lambda connection: connection.execute(
            "UPDATE local_observation_fragments SET body_json = '{' "
            "WHERE content_kind = 'llm_response'"
        )
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await repository.get_llm_call(SCOPE, "call-corrupt")
    await database.close()


def test_new_public_observation_api_docstrings_follow_repository_format() -> None:
    methods = tuple(
        member
        for name, member in inspect.getmembers(LocalObservationRepository, inspect.isfunction)
        if not name.startswith("_")
    ) + (LocalSQLiteDatabase.read_transaction,)
    for method in methods:
        docstring = inspect.getdoc(method) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2
