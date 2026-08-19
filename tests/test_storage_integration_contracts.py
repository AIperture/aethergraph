from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    ExternalSessionBindingRecord,
    ExternalSessionBindingRequest,
    IngressClaimRecord,
    IngressClaimRequest,
    IngressClaimResult,
    IngressClaimStatus,
    IngressIdempotencyRepository,
    IntegrationSessionProvisioningResult,
    IntegrationSessionRepository,
    SessionKind,
    SessionRecord,
    StorageBundle,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
HOST_SCOPE = StorageScope(tenant_id="tenant-1", project_id="project-1")
EXTERNAL_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    scope_key='{"conversation_id":"C1","thread_id":"T1"}',
)


def _claim_request() -> IngressClaimRequest:
    return IngressClaimRequest(
        deployment_id="deployment-1",
        integration_id="slack-main",
        idempotency_key="event-1",
        external_event_id="event-1",
        envelope_digest="a" * 64,
        digest_algorithm="sha256",
        scope=HOST_SCOPE,
        claimed_at=NOW,
    )


def _pending_claim() -> IngressClaimRecord:
    request = _claim_request()
    return IngressClaimRecord(
        deployment_id=request.deployment_id,
        integration_id=request.integration_id,
        idempotency_key=request.idempotency_key,
        external_event_id=request.external_event_id,
        envelope_digest=request.envelope_digest,
        digest_algorithm=request.digest_algorithm,
        scope=request.scope,
        claimed_at=request.claimed_at,
        status=IngressClaimStatus.PENDING,
        revision=1,
    )


def test_ingress_claims_are_scoped_revisioned_and_single_assignment() -> None:
    pending = _pending_claim()
    completed = replace(
        pending,
        status=IngressClaimStatus.COMPLETED,
        revision=2,
        receipt={"accepted": True, "action": "root_turn_started"},
        completed_at=NOW + timedelta(seconds=1),
    )

    assert IngressClaimResult(record=pending, acquired=True).acquired
    assert completed.receipt["accepted"] is True
    assert "app_id" not in {item.name for item in fields(IngressClaimRecord)}
    assert "client_id" not in {item.name for item in fields(IngressClaimRequest)}
    with pytest.raises(ValueError, match="must not have a receipt"):
        replace(pending, receipt={"accepted": True})
    with pytest.raises(ValueError, match="only a pending"):
        IngressClaimResult(record=completed, acquired=True)


def test_ingress_receipts_are_deeply_immutable() -> None:
    receipt = {"outputs": ["one"]}
    completed = replace(
        _pending_claim(),
        status=IngressClaimStatus.COMPLETED,
        revision=2,
        receipt=receipt,
        completed_at=NOW,
    )
    receipt["outputs"].append("two")

    assert completed.receipt["outputs"] == ("one",)


def _binding_request() -> ExternalSessionBindingRequest:
    return ExternalSessionBindingRequest(
        binding_id="binding-1",
        route_id="route-1",
        build_id="build-1",
        ag_session_id="session-1",
        scope=EXTERNAL_SCOPE,
        now=NOW,
    )


def test_external_bindings_require_opaque_scope_and_exclude_host_dtos() -> None:
    request = _binding_request()
    record = ExternalSessionBindingRecord(
        binding_id=request.binding_id,
        route_id=request.route_id,
        build_id=request.build_id,
        ag_session_id=request.ag_session_id,
        scope=request.scope,
        revision=1,
        created_at=NOW,
        last_seen_at=NOW,
        metadata={"integration_kind": "slack"},
    )

    session = SessionRecord(
        session_id="session-1",
        kind=SessionKind.CHAT,
        scope=StorageScope(project_id="project-1", session_id="session-1"),
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
    result = IntegrationSessionProvisioningResult(
        session=session,
        binding=record,
        session_created=True,
        binding_created=True,
    )
    assert result.session_created and result.binding_created
    assert record.metadata["integration_kind"] == "slack"
    assert "external_identity" not in {item.name for item in fields(record)}
    assert "route" not in {item.name for item in fields(request)}
    with pytest.raises(ValueError, match="scope_key"):
        replace(request, scope=HOST_SCOPE)
    with pytest.raises(ValueError, match="run_id or node_id"):
        replace(request, scope=replace(EXTERNAL_SCOPE, run_id="run-1"))


def test_integration_bundle_fields_and_protocol_docstrings_are_exact() -> None:
    hints = get_type_hints(StorageBundle)
    assert hints["ingress_idempotency"] is IngressIdempotencyRepository
    assert hints["integration_sessions"] is IntegrationSessionRepository

    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for protocol in (IngressIdempotencyRepository, IntegrationSessionRepository):
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)
