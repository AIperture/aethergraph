from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime
import inspect
from typing import get_type_hints

import pytest

from aethergraph.contracts.integration import SemanticEventKind as HostSemanticEventKind
from aethergraph.storage.contracts import (
    InboundEventDraft,
    InboundEventRecord,
    InboundEventRepository,
    RuntimeOutputFrame,
    RuntimeOutputSink,
    RuntimeOutputStream,
    SemanticEventDraft,
    SemanticEventKind,
    SemanticEventQuery,
    SemanticEventRecord,
    SemanticEventRepository,
    StorageBundle,
    StorageCapacityError,
    StorageError,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
SESSION_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
)
RUN_SCOPE = replace(
    SESSION_SCOPE,
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
)


def test_inbound_events_are_closed_scoped_and_deeply_immutable() -> None:
    payload = {"text": "hello", "attachments": ["artifact:one"]}
    draft = InboundEventDraft(
        event_id="ingress-1",
        deployment_id="deployment-1",
        route_id="route-1",
        integration_id="slack-main",
        external_event_id="event-1",
        received_at=NOW,
        scope=SESSION_SCOPE,
        payload=payload,
        resource_keys=("artifact:one",),
    )
    record = InboundEventRecord(
        event_id=draft.event_id,
        deployment_id=draft.deployment_id,
        route_id=draft.route_id,
        integration_id=draft.integration_id,
        external_event_id=draft.external_event_id,
        received_at=draft.received_at,
        scope=draft.scope,
        delivery_cursor=1,
        cursor="cursor-1",
        payload=draft.payload,
        resource_keys=draft.resource_keys,
    )
    payload["attachments"].append("artifact:two")

    assert draft.payload["attachments"] == ("artifact:one",)
    assert record.delivery_cursor == 1
    assert record.cursor == "cursor-1"
    assert "app_id" not in {item.name for item in fields(InboundEventRecord)}
    with pytest.raises(ValueError, match="delivery_cursor"):
        replace(record, delivery_cursor=0)
    with pytest.raises(ValueError, match="session_id"):
        replace(draft, scope=StorageScope(tenant_id="tenant-1"))


def _semantic_draft() -> SemanticEventDraft:
    return SemanticEventDraft(
        event_id="semantic-1",
        deployment_id="deployment-1",
        turn_id="turn-1",
        sequence=0,
        producer="agent.support",
        occurred_at=NOW,
        kind=SemanticEventKind.MESSAGE_COMPLETED,
        scope=SESSION_SCOPE,
        payload={"message_id": "message-1", "text": "done"},
    )


def test_semantic_events_preserve_authored_sequence_and_opaque_cursor() -> None:
    draft = _semantic_draft()
    record = SemanticEventRecord(
        event_id=draft.event_id,
        deployment_id=draft.deployment_id,
        turn_id=draft.turn_id,
        sequence=draft.sequence,
        producer=draft.producer,
        occurred_at=draft.occurred_at,
        kind=draft.kind,
        scope=draft.scope,
        delivery_cursor=2,
        cursor="cursor-2",
        payload=draft.payload,
    )
    query = SemanticEventQuery(
        deployment_id="deployment-1",
        scope=SESSION_SCOPE,
        kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
    )

    assert record.sequence == 0
    assert record.delivery_cursor == 2
    assert query.page.limit > 0
    with pytest.raises(ValueError, match="delivery_cursor"):
        replace(record, delivery_cursor=True)
    with pytest.raises(ValueError, match="non-negative"):
        replace(draft, sequence=-1)
    with pytest.raises(ValueError, match="duplicates"):
        replace(query, kinds=(draft.kind, draft.kind))


def test_semantic_event_kind_matches_exact_active_host_v2_vocabulary() -> None:
    assert {kind.value for kind in SemanticEventKind} == {
        kind.value for kind in HostSemanticEventKind
    }
    assert "run.status_changed" not in SemanticEventKind
    assert "error" not in SemanticEventKind


def test_runtime_output_frames_require_canonical_run_node_scope() -> None:
    frame = RuntimeOutputFrame(
        output_id="output-1",
        execution_id="execution-1",
        scope=RUN_SCOPE,
        stream=RuntimeOutputStream.STDOUT,
        sequence=1,
        text="hello",
        source="python",
        tags=("runtime-console",),
    )

    assert frame.scope.run_id == "run-1"
    assert issubclass(StorageCapacityError, StorageError)
    assert "app_id" not in {item.name for item in fields(RuntimeOutputFrame)}
    with pytest.raises(ValueError, match="positive"):
        replace(frame, sequence=0)
    with pytest.raises(TypeError, match="RuntimeOutputStream"):
        replace(frame, stream="stdout")


def test_stream_bundle_fields_and_protocol_docstrings_are_exact() -> None:
    hints = get_type_hints(StorageBundle)
    assert hints["inbound_events"] is InboundEventRepository
    assert hints["semantic_events"] is SemanticEventRepository
    assert hints["runtime_output"] is RuntimeOutputSink

    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for protocol in (InboundEventRepository, SemanticEventRepository, RuntimeOutputSink):
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)
