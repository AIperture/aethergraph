from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    LLMCallAttempt,
    LLMCallDetail,
    LLMCallDraft,
    LLMCallLifecycleStatus,
    LLMCallQuery,
    LLMCallRecord,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationLLMSummaryQuery,
    ObservationLLMSummaryRecord,
    ObservationPurgeRequest,
    ObservationPurgeResult,
    ObservationQuery,
    ObservationRecord,
    ObservationRepository,
    ObservationResourceLink,
    ObservationResourceRelation,
    ObservationScopeManagementQuery,
    ObservationScopeManagementRecord,
    ObservationScopeUsageQuery,
    ObservationScopeUsageRecord,
    ObservationSeverity,
    ObservationStatus,
    ObservationStorageStats,
    ObservationTraceSummaryQuery,
    ObservationTraceSummaryRecord,
    ObservationUsageDimension,
    StorageBundle,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
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
)
LINK = ObservationResourceLink(
    resource_key="artifact:report",
    relation=ObservationResourceRelation.OUTPUT,
    resource_revision="2",
)


def _observation(*, category: str = "trace") -> ObservationDraft:
    return ObservationDraft(
        observation_id="obs-1",
        category=category,
        name="runner.execute",
        summary="runner finished",
        occurred_at=NOW,
        scope=SCOPE,
        producer="aethergraph.runtime",
        trace_id="trace-1",
        attributes={"metrics": {"duration_ms": 12}, "tags": ["runtime"]},
        resource_links=(LINK,),
    )


def _stored_observation(*, category: str = "trace") -> ObservationRecord:
    draft = _observation(category=category)
    return ObservationRecord(
        observation_id=draft.observation_id,
        category=draft.category,
        name=draft.name,
        summary=draft.summary,
        occurred_at=draft.occurred_at,
        scope=draft.scope,
        cursor="cursor-1",
        trace_id=draft.trace_id,
        attributes=draft.attributes,
        resource_links=draft.resource_links,
        producer=draft.producer,
    )


def test_observation_records_are_scoped_ordered_and_deeply_immutable() -> None:
    attributes = {"tags": ["runtime"]}
    draft = replace(_observation(), attributes=attributes)
    attributes["tags"].append("changed")

    assert draft.attributes["tags"] == ("runtime",)
    assert _stored_observation().cursor == "cursor-1"
    assert draft.producer == "aethergraph.runtime"
    assert "app_id" not in {item.name for item in fields(ObservationRecord)}
    assert "application_id" not in {item.name for item in fields(ObservationDraft)}
    with pytest.raises(ValueError, match="duplicate identities"):
        replace(draft, resource_links=(LINK, LINK))


def test_observation_query_is_bounded_and_resource_indexed() -> None:
    query = ObservationQuery(
        scope=SCOPE,
        categories=("trace",),
        names=("runner.execute",),
        producers=("aethergraph.runtime",),
        statuses=(ObservationStatus.ERROR,),
        severities=(ObservationSeverity.ERROR,),
        trace_id="trace-1",
        resource_key="artifact:report",
        resource_relation=ObservationResourceRelation.OUTPUT,
        occurred_at_or_after=NOW - timedelta(minutes=1),
        occurred_at_or_before=NOW,
    )

    assert query.page.limit > 0
    with pytest.raises(ValueError, match="requires resource_key"):
        replace(query, resource_key=None)
    with pytest.raises(ValueError, match="time bounds are reversed"):
        replace(query, occurred_at_or_after=NOW + timedelta(seconds=1))


def test_llm_call_records_separate_list_metadata_from_full_content() -> None:
    observation = replace(
        _observation(category="llm"),
        name="chat",
        summary="openai/model chat",
    )
    attempt = LLMCallAttempt(
        attempt_number=1,
        elapsed_ms=100,
        outcome="success",
        retryable=False,
        status_code=200,
        rate_limits=({"remaining": 10},),
    )
    draft = LLMCallDraft(
        llm_call_id="call-1",
        observation=observation,
        call_type="chat",
        provider="openai",
        model="model",
        capture_mode=ObservationCaptureMode.FULL,
        prompt_manifest_id="manifest-1",
        request_options={"temperature": 0.2},
        usage={"input_tokens": 10},
        request_preview={"messages": 1},
        captured_request={"messages": [{"role": "user", "content": "hello"}]},
        captured_response={"text": "hi"},
        attempts=(attempt,),
    )
    stored_observation = replace(_stored_observation(category="llm"), name="chat")
    record = LLMCallRecord(
        llm_call_id="call-1",
        observation=stored_observation,
        call_type="chat",
        provider="openai",
        model="model",
        capture_mode=ObservationCaptureMode.FULL,
        prompt_manifest_id="manifest-1",
        request_options=draft.request_options,
        usage=draft.usage,
        request_preview=draft.request_preview,
        attempts=(attempt,),
    )
    detail = LLMCallDetail(
        record=record,
        captured_request=draft.captured_request,
        captured_response=draft.captured_response,
    )

    assert record.request_preview["messages"] == 1
    assert detail.captured_request["messages"][0]["content"] == "hello"
    assert "captured_request" not in {item.name for item in fields(LLMCallRecord)}
    with pytest.raises(ValueError, match="only manifest or full capture"):
        LLMCallDetail(
            record=replace(record, capture_mode=ObservationCaptureMode.METADATA),
            captured_request={},
        )


def test_llm_queries_attempts_and_errors_fail_closed() -> None:
    query = LLMCallQuery(
        scope=SCOPE,
        providers=("openai",),
        models=("model",),
        statuses=(ObservationStatus.ERROR,),
    )

    assert query.page.limit > 0
    with pytest.raises(ValueError, match="contiguous"):
        replace(
            LLMCallDraft(
                llm_call_id="call-1",
                observation=replace(
                    _observation(category="llm"),
                    status=ObservationStatus.ERROR,
                ),
                call_type="chat",
                provider="openai",
                model="model",
                capture_mode=ObservationCaptureMode.METADATA,
                lifecycle_status=LLMCallLifecycleStatus.FAILED,
                prompt_manifest_id="manifest-1",
                error_type="RateLimit",
                error_message="retry",
            ),
            attempts=(
                LLMCallAttempt(
                    attempt_number=2,
                    elapsed_ms=1,
                    outcome="error",
                    retryable=True,
                ),
            ),
        )


def test_observation_summary_contracts_are_bounded_truthful_and_immutable() -> None:
    trace_query = ObservationTraceSummaryQuery(
        scope=SCOPE,
        occurred_at_or_after=NOW - timedelta(minutes=1),
        occurred_at_or_before=NOW,
        trace_id_limit=25,
        failing_service_limit=3,
    )
    trace_summary = ObservationTraceSummaryRecord(
        span_count=4,
        error_count=2,
        total_duration_ms=30,
        trace_id_count=3,
        trace_ids=("trace-1", "trace-2"),
        trace_ids_truncated=True,
        top_failing_services={"runner": 2},
        latest_error_at=NOW,
    )
    llm_query = ObservationLLMSummaryQuery(scope=SCOPE, model_limit=10)
    llm_summary = ObservationLLMSummaryRecord(
        total_calls=5,
        total_prompt_tokens=20,
        total_completion_tokens=10,
        total_tokens=30,
        error_count=1,
        model_count=2,
        by_model={"model-a": 4},
        by_model_truncated=True,
    )

    assert trace_query.trace_id_limit == 25
    assert trace_summary.trace_ids_truncated
    assert llm_query.model_limit == 10
    assert llm_summary.by_model_truncated
    with pytest.raises(TypeError):
        trace_summary.top_failing_services["other"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="between 1 and 500"):
        replace(trace_query, trace_id_limit=501)
    with pytest.raises(ValueError, match="must match trace_id_count"):
        replace(trace_summary, trace_ids_truncated=False)
    with pytest.raises(ValueError, match="must match model_count"):
        replace(llm_summary, by_model_truncated=False)


def test_retention_records_are_bounded_revisioned_and_provider_neutral() -> None:
    request = ObservationPurgeRequest(
        scope=SCOPE,
        dry_run=True,
        capture_modes=(ObservationCaptureMode.FULL,),
        retention_classes=("forensic",),
        excluded_severities=(ObservationSeverity.ERROR,),
        occurred_before=NOW,
        max_observations=100,
        target_reclaimed_bytes=1024,
    )
    preview = ObservationPurgeResult(
        dry_run=True,
        matching_traces=1,
        matching_observations=2,
        matching_manifests=1,
        exclusive_fragment_bytes=100,
        shared_fragment_bytes_retained=50,
        estimated_reclaimed_bytes=500,
    )
    stats = ObservationStorageStats(
        observations=2,
        llm_calls=1,
        manifests=1,
        fragments=2,
        fragment_bytes=150,
        logical_bytes=500,
        provider_metrics={"allocated_bytes": 4096},
    )
    policy = ObservationScopeManagementRecord(
        scope_key="trace:trace-1",
        scope=SCOPE,
        trace_id="trace-1",
        revision=1,
        updated_at=NOW,
        pinned=True,
        tags=("important",),
    )
    usage_query = ObservationScopeUsageQuery(
        scope=SCOPE,
        dimension=ObservationUsageDimension.TRACE,
    )
    usage = ObservationScopeUsageRecord(
        dimension=ObservationUsageDimension.TRACE,
        scope_id="trace-1",
        latest_at=NOW,
        observation_count=2,
        logical_bytes=500,
        pinned=True,
    )
    management_query = ObservationScopeManagementQuery(
        scope=SCOPE,
        pinned=True,
        retention_classes=("standard",),
    )

    assert request.dry_run and preview.deleted_observations == 0
    assert stats.provider_metrics["allocated_bytes"] == 4096
    assert policy.pinned
    assert usage_query.page.limit > 0 and usage.logical_bytes == 500
    assert management_query.pinned is True
    with pytest.raises(ValueError, match="must not overlap"):
        replace(
            request,
            severities=(ObservationSeverity.ERROR,),
            excluded_severities=(ObservationSeverity.ERROR,),
        )
    with pytest.raises(ValueError, match="dry-run"):
        replace(preview, deleted_observations=1)


def test_observation_bundle_field_and_protocol_docstrings_are_exact() -> None:
    assert get_type_hints(StorageBundle)["observations"] is ObservationRepository

    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for name, member in inspect.getmembers(ObservationRepository, inspect.isfunction):
        if name.startswith("_"):
            continue
        docstring = inspect.getdoc(member) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), name
        assert positions == tuple(sorted(positions)), name
