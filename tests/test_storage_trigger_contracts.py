from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    ClaimedTrigger,
    StorageBundle,
    StorageScope,
    TriggerClaimRecord,
    TriggerClaimRequest,
    TriggerClaimStatus,
    TriggerKind,
    TriggerQuery,
    TriggerRecord,
    TriggerRepository,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
    graph_id="graph-1",
)


def _trigger() -> TriggerRecord:
    return TriggerRecord(
        trigger_id="trigger-1",
        graph_id="graph-1",
        scope=SCOPE,
        kind=TriggerKind.INTERVAL,
        interval_seconds=60,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        next_fire_at=NOW + timedelta(minutes=1),
        default_inputs={"messages": ["hello"]},
        metadata={"source": "test"},
    )


def test_trigger_record_is_scoped_revisioned_and_deeply_immutable() -> None:
    inputs = {"messages": ["hello"]}
    record = replace(_trigger(), default_inputs=inputs)
    inputs["messages"].append("changed")

    assert record.default_inputs["messages"] == ("hello",)
    assert "app_id" not in {item.name for item in fields(TriggerRecord)}
    assert "client_id" not in {item.name for item in fields(TriggerRecord)}
    with pytest.raises(ValueError, match="run_id or node_id"):
        replace(record, scope=replace(SCOPE, run_id="run-1"))


def test_trigger_kind_configuration_fails_closed() -> None:
    record = _trigger()

    with pytest.raises(ValueError, match="must not define interval_seconds"):
        replace(record, kind=TriggerKind.EVENT, event_key="invoice.paid", next_fire_at=None)
    with pytest.raises(ValueError, match="active scheduled triggers"):
        replace(record, next_fire_at=None)
    with pytest.raises(TypeError, match="TriggerKind"):
        replace(record, kind="interval")


def test_trigger_queries_are_bounded_and_event_explicit() -> None:
    query = TriggerQuery(
        scope=SCOPE,
        kinds=(TriggerKind.EVENT,),
        active=True,
        event_key="invoice.paid",
    )

    assert query.page.limit > 0
    with pytest.raises(ValueError, match="event trigger kind"):
        replace(query, kinds=(TriggerKind.CRON,))
    with pytest.raises(ValueError, match="duplicates"):
        replace(query, kinds=(TriggerKind.EVENT, TriggerKind.EVENT))
    with pytest.raises(ValueError, match="explicit canonical scope"):
        replace(query, scope=StorageScope())


def test_trigger_claim_request_and_lifecycle_are_revision_safe() -> None:
    request = TriggerClaimRequest(
        now=NOW,
        worker_id="worker-a",
        lease_until=NOW + timedelta(seconds=30),
        limit=100,
    )
    leased = TriggerClaimRecord(
        fire_id="fire-1",
        trigger_id="trigger-1",
        scope=SCOPE,
        scheduled_for=NOW,
        status=TriggerClaimStatus.LEASED,
        attempts=1,
        revision=1,
        updated_at=NOW,
        worker_id="worker-a",
        lease_until=NOW + timedelta(seconds=30),
    )
    delivered = replace(
        leased,
        status=TriggerClaimStatus.DELIVERED,
        revision=2,
        updated_at=NOW + timedelta(seconds=1),
        worker_id=None,
        lease_until=None,
        run_id="run-1",
        finished_at=NOW + timedelta(seconds=1),
    )

    assert ClaimedTrigger(trigger=_trigger(), claim=leased).claim == leased
    assert delivered.run_id == "run-1"
    with pytest.raises(ValueError, match="between 1 and 1000"):
        replace(request, limit=0)
    with pytest.raises(ValueError, match="delivered claims require"):
        replace(delivered, run_id=None)


def test_trigger_bundle_field_and_protocol_docstrings_are_exact() -> None:
    assert get_type_hints(StorageBundle)["triggers"] is TriggerRepository

    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for name, member in inspect.getmembers(TriggerRepository, inspect.isfunction):
        if name.startswith("_"):
            continue
        docstring = inspect.getdoc(member) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), name
        assert positions == tuple(sorted(positions)), name
