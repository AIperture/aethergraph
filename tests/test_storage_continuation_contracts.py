from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    ContinuationCorrelator,
    ContinuationDraft,
    ContinuationLeaseQuery,
    ContinuationLeaseRecord,
    ContinuationLeaseRepository,
    ContinuationLeaseRequest,
    ContinuationLeaseStatus,
    ContinuationQuery,
    ContinuationRecord,
    ContinuationRepository,
    ContinuationStatus,
    CreatedContinuation,
    StorageBundle,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
    node_id="approval",
)
CORRELATOR = ContinuationCorrelator(
    scheme="slack",
    channel="C123",
    thread="T123",
    message="M123",
)


def _waiting_record() -> ContinuationRecord:
    return ContinuationRecord(
        continuation_id="cont-1",
        kind="approval",
        scope=SCOPE,
        created_at=NOW,
        token_digest="sha256:digest",
        revision=1,
        prompt="Approve?",
        resume_schema={"type": "object", "required": ["approved"]},
        payload={"context": ["a"]},
        deadline=NOW + timedelta(hours=1),
        next_wakeup_at=NOW + timedelta(minutes=5),
        channel="ui:session",
        correlators=(CORRELATOR,),
    )


def test_continuation_draft_and_record_are_scoped_and_deeply_immutable() -> None:
    payload = {"context": ["a"]}
    draft = ContinuationDraft(
        continuation_id="cont-1",
        kind="approval",
        scope=SCOPE,
        created_at=NOW,
        payload=payload,
        correlators=(CORRELATOR,),
    )
    record = _waiting_record()
    payload["context"].append("b")

    assert draft.payload["context"] == ("a",)
    assert record.resume_schema["required"] == ("approved",)
    assert "token" not in {item.name for item in fields(ContinuationRecord)}
    assert "app_id" not in {item.name for item in fields(ContinuationRecord)}
    assert "application_id" not in {item.name for item in fields(ContinuationDraft)}


def test_continuation_lifecycle_and_creation_result_fail_closed() -> None:
    waiting = _waiting_record()
    resumed = replace(
        waiting,
        status=ContinuationStatus.RESUMED,
        revision=2,
        closed_at=NOW + timedelta(minutes=1),
    )

    assert CreatedContinuation(record=waiting, token="raw-secret").token == "raw-secret"
    assert resumed.status is ContinuationStatus.RESUMED
    with pytest.raises(ValueError, match="terminal continuations require"):
        replace(waiting, status=ContinuationStatus.CANCELED)
    with pytest.raises(ValueError, match="created continuation must be waiting"):
        CreatedContinuation(record=resumed, token="raw-secret")
    with pytest.raises(ValueError, match="deadline must not precede"):
        replace(waiting, deadline=NOW - timedelta(seconds=1))
    with pytest.raises(TypeError, match="ContinuationStatus"):
        replace(waiting, status="waiting")


def test_continuation_queries_are_bounded_and_index_explicit() -> None:
    query = ContinuationQuery(
        scope=SCOPE,
        statuses=(ContinuationStatus.WAITING,),
        kinds=("approval",),
        channel="ui:session",
        correlator=CORRELATOR,
        due_at_or_before=NOW,
    )

    assert query.correlator == CORRELATOR
    assert query.page.limit > 0
    with pytest.raises(ValueError, match="duplicates"):
        replace(query, statuses=(ContinuationStatus.WAITING, ContinuationStatus.WAITING))


def test_lease_request_and_records_enforce_ownership_state() -> None:
    request = ContinuationLeaseRequest(
        fire_id="fire-1",
        continuation_id="cont-1",
        scope=SCOPE,
        scheduled_for=NOW,
        worker_id="worker-a",
        now=NOW,
        lease_until=NOW + timedelta(seconds=30),
    )
    leased = ContinuationLeaseRecord(
        fire_id=request.fire_id,
        continuation_id=request.continuation_id,
        scope=request.scope,
        scheduled_for=request.scheduled_for,
        status=ContinuationLeaseStatus.LEASED,
        attempts=1,
        revision=1,
        updated_at=NOW,
        worker_id="worker-a",
        lease_until=NOW + timedelta(seconds=30),
    )
    delivered = replace(
        leased,
        status=ContinuationLeaseStatus.DELIVERED,
        revision=2,
        updated_at=NOW + timedelta(seconds=1),
        worker_id=None,
        lease_until=None,
        finished_at=NOW + timedelta(seconds=1),
    )

    assert delivered.status is ContinuationLeaseStatus.DELIVERED
    assert "token" not in {item.name for item in fields(ContinuationLeaseRecord)}
    with pytest.raises(ValueError, match="lease_until must be after now"):
        replace(request, lease_until=NOW)
    with pytest.raises(ValueError, match="scheduled_for must not be after now"):
        replace(request, scheduled_for=NOW + timedelta(seconds=1))
    with pytest.raises(ValueError, match="retry records require"):
        replace(
            delivered,
            status=ContinuationLeaseStatus.RETRY,
            finished_at=None,
        )


def test_lease_query_and_bundle_use_exact_protocols() -> None:
    query = ContinuationLeaseQuery(
        scope=SCOPE,
        statuses=(ContinuationLeaseStatus.DEAD_LETTER,),
        continuation_id="cont-1",
    )
    hints = get_type_hints(StorageBundle)

    assert query.page.limit > 0
    assert hints["continuations"] is ContinuationRepository
    assert hints["continuation_leases"] is ContinuationLeaseRepository


def test_continuation_protocol_docstrings_follow_required_section_order() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for protocol in (ContinuationRepository, ContinuationLeaseRepository):
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)
