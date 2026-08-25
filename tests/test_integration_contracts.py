from datetime import UTC, datetime
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from aethergraph.contracts.integration import (
    INTEGRATION_CAPABILITIES_SCHEMA_VERSION,
    INTEGRATION_ROUTE_SCHEMA_VERSION,
    RELEASE_COMPATIBILITY_SCHEMA_VERSION,
    SEMANTIC_EVENT_PROTOCOL_VERSION,
    AcceptedEventContract,
    AgentInputV1,
    ExternalIdentity,
    HostManifest,
    IngressEnvelope,
    IngressReceipt,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    MessageCompletedPayload,
    OriginAddress,
    SemanticEvent,
    SemanticEventKind,
    ToolActivityPayload,
    ToolErrorPayload,
    TurnOutcomePayload,
)
from tests._integration_fixtures import contract_compatibility

_DIGEST = "a" * 64
_PROTOCOL_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "integration" / "semantic_event_v3.json"
)


def _capabilities() -> IntegrationCapabilities:
    return IntegrationCapabilities(
        event_kinds=(
            SemanticEventKind.MESSAGE_STARTED,
            SemanticEventKind.MESSAGE_DELTA,
            SemanticEventKind.MESSAGE_COMPLETED,
            SemanticEventKind.TURN_OUTCOME,
        ),
        streaming=True,
        interactions=False,
        attachments=False,
        cancellation=True,
    )


def _route() -> IntegrationRoute:
    return IntegrationRoute(
        route_id="route_ui",
        endpoint_id="ui",
        integration_id="integration_ui",
        integration_kind=IntegrationKind.AG_UI,
        entry_agent_id="agent.support",
        enabled=True,
        match_policy=IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope="conversation"),
        required_capabilities=_capabilities(),
    )


def _manifest() -> HostManifest:
    return HostManifest(
        deployment_id="deployment_1",
        build_id="build_1",
        source_digest=_DIGEST,
        build_root="C:/workspace/build_1",
        entrypoint_module="compiled_support.runtime",
        entrypoint_symbol="register",
        graph_id="graph.support",
        entry_agent_id="agent.support",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        release_compatibility=contract_compatibility(),
        integration_routes=(_route(),),
        logical_output_bindings={"primary": "origin"},
        workspace_identity="workspace_1",
        manifest_digest=_DIGEST,
    )


def test_host_manifest_is_closed_and_round_trips_across_json_boundary() -> None:
    manifest = _manifest()

    restored = HostManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert restored.integration_routes[0].entry_agent_id == "agent.support"


def test_containing_contracts_use_coordinated_negotiation_schema_versions() -> None:
    manifest = _manifest()

    assert manifest.schema_version == "aethergraph.host-manifest/v4"
    assert manifest.release_compatibility.schema_version == (RELEASE_COMPATIBILITY_SCHEMA_VERSION)
    assert manifest.integration_routes[0].schema_version == INTEGRATION_ROUTE_SCHEMA_VERSION
    assert manifest.integration_routes[0].required_capabilities.schema_version == (
        INTEGRATION_CAPABILITIES_SCHEMA_VERSION
    )
    assert SEMANTIC_EVENT_PROTOCOL_VERSION == "aethergraph.semantic-event/v3"


def test_manifest_accepts_only_the_canonical_semantic_protocol() -> None:
    manifest = _manifest()

    assert manifest.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_VERSION
    assert manifest.integration_routes[0].required_capabilities.event_kinds[-1] == (
        SemanticEventKind.TURN_OUTCOME
    )
    assert all(
        isinstance(kind, SemanticEventKind)
        for kind in manifest.integration_routes[0].required_capabilities.event_kinds
    )


@pytest.mark.parametrize(
    ("path", "legacy_version"),
    [
        (("schema_version",), "aethergraph.host-manifest/v2"),
        (
            ("release_compatibility", "schema_version"),
            "aethergraph.release-compatibility/v1",
        ),
        (
            ("integration_routes", 0, "schema_version"),
            "aethergraph.integration-route/v1",
        ),
        (
            (
                "integration_routes",
                0,
                "required_capabilities",
                "schema_version",
            ),
            "aethergraph.integration-capabilities/v1",
        ),
    ],
)
def test_containing_contracts_reject_superseded_schema_versions(
    path: tuple[str | int, ...],
    legacy_version: str,
) -> None:
    payload = _manifest().model_dump(mode="json")
    target: object = payload
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = legacy_version  # type: ignore[index]

    with pytest.raises(ValidationError):
        HostManifest.model_validate(payload)


def test_capabilities_reject_superseded_protocol_and_event_kinds() -> None:
    with pytest.raises(ValidationError):
        IntegrationCapabilities(
            semantic_event_protocol_version="aethergraph.semantic-event/v1",
            event_kinds=("turn.completed",),
            streaming=False,
            interactions=False,
            attachments=False,
            cancellation=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "aethergraph.host-manifest/v1"),
        ("ingress_protocol_version", "aethergraph.ingress/v1"),
        ("semantic_event_protocol_version", "aethergraph.semantic-event/v1"),
    ],
)
def test_host_manifest_rejects_unknown_contract_versions(field: str, value: str) -> None:
    payload = _manifest().model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError):
        HostManifest.model_validate(payload)


def test_host_manifest_rejects_extra_fields_and_ambiguous_sources() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["bot_token"] = "must-never-enter-a-manifest"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HostManifest.model_validate(payload)

    payload.pop("bot_token")
    payload["release_uri"] = "https://example.invalid/release.zip"
    with pytest.raises(ValidationError, match="exactly one"):
        HostManifest.model_validate(payload)


def test_route_requires_endpoint_identity_only_for_endpoint_transports() -> None:
    payload = _route().model_dump(mode="json")
    payload["integration_kind"] = "slack"

    with pytest.raises(ValidationError, match="endpoint_id is required only"):
        IntegrationRoute.model_validate(payload)

    payload["endpoint_id"] = None
    slack_route = IntegrationRoute.model_validate(payload)
    assert slack_route.integration_kind is IntegrationKind.SLACK


def test_ingress_envelope_is_closed_and_requires_one_command() -> None:
    envelope = IngressEnvelope(
        integration_id="integration_ui",
        endpoint_id="ui",
        external_identity=ExternalIdentity(
            tenant_id="tenant_1",
            conversation_id="conversation_1",
            user_id="user_1",
        ),
        external_event_id="event_1",
        idempotency_key="event_1",
        received_at=datetime(2026, 8, 3, tzinfo=UTC),
        input=AgentInputV1(
            input_id="event_1",
            kind="message",
            type="user.message",
            source="urn:test:ui",
            occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
            payload={"text": "Hello"},
        ),
        origin_address=OriginAddress(
            channel_key="endpoint:sessions/session_1",
            capability_profile_id="ag_ui_v1",
        ),
    )

    restored = IngressEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope

    payload = envelope.model_dump(mode="json")
    payload["external_event_id"] = "another-event"
    with pytest.raises(ValidationError, match="must match input.input_id"):
        IngressEnvelope.model_validate(payload)

    payload["external_event_id"] = "event_1"
    payload["raw_provider_payload"] = {"unbounded": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IngressEnvelope.model_validate(payload)


def test_agent_input_round_trips_message_and_event_without_inferred_kind() -> None:
    message = AgentInputV1(
        input_id="message_1",
        kind="message",
        type="user.message",
        source="urn:test:studio",
        occurred_at=datetime(2026, 8, 23, tzinfo=UTC),
        payload={"text": "Review this design."},
    )
    event = AgentInputV1(
        input_id="event_1",
        kind="event",
        type="simulation.abnormal",
        source="urn:test:simulator",
        occurred_at=datetime(2026, 8, 23, tzinfo=UTC),
        subject="wafer-run-42",
        payload={"metric": "focus_error_nm", "value": 18.4},
    )

    assert AgentInputV1.model_validate_json(message.model_dump_json()) == message
    assert AgentInputV1.model_validate_json(event.model_dump_json()) == event

    with pytest.raises(ValidationError, match="reserved message type"):
        AgentInputV1.model_validate({**event.model_dump(mode="json"), "type": "user.message"})


def test_accepted_event_contract_validates_schema_example_and_manifest_match() -> None:
    contract = AcceptedEventContract(
        type="simulation.abnormal",
        title="Abnormal simulation",
        payload_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        example_payload={"value": 18.4},
    )
    compatibility = contract_compatibility().model_copy(update={"accepted_events": (contract,)})
    manifest = _manifest().model_copy(
        update={
            "release_compatibility": compatibility,
            "accepted_events": (contract,),
        }
    )

    assert HostManifest.model_validate_json(manifest.model_dump_json()) == manifest

    with pytest.raises(ValidationError, match="example_payload does not match"):
        AcceptedEventContract(
            type="simulation.abnormal",
            title="Abnormal simulation",
            payload_schema={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
            example_payload={"value": "not-a-number"},
        )

    with pytest.raises(ValidationError, match="must match release compatibility"):
        HostManifest.model_validate(
            manifest.model_copy(update={"accepted_events": ()}).model_dump(mode="json")
        )


def test_receipt_enforces_terminal_result_shape() -> None:
    accepted = IngressReceipt(
        accepted=True,
        duplicate=False,
        action="root_turn_started",
        deployment_id="deployment_1",
        route_id="route_ui",
        session_id="session_1",
        turn_id="turn_1",
        event_cursor=1,
    )
    assert accepted.rejection_code is None
    assert accepted.rejection_message is None

    with pytest.raises(ValidationError, match="accepted receipts cannot include rejection"):
        IngressReceipt(
            accepted=True,
            duplicate=False,
            action="root_turn_started",
            deployment_id="deployment_1",
            route_id="route_ui",
            session_id="session_1",
            turn_id="turn_1",
            event_cursor=1,
            rejection_message="This must not be present.",
        )

    with pytest.raises(ValidationError, match="rejected receipts require"):
        IngressReceipt(
            accepted=False,
            duplicate=False,
            action="rejected",
            deployment_id="deployment_1",
        )


def test_semantic_event_parses_exact_payload_and_rejects_mismatch() -> None:
    event = SemanticEvent(
        event_id="event_1",
        deployment_id="deployment_1",
        session_id="session_1",
        turn_id="turn_1",
        sequence=3,
        producer="agent.support",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        kind=SemanticEventKind.MESSAGE_COMPLETED,
        payload=MessageCompletedPayload(
            message_id="message_1",
            text="Done",
        ),
        extensions={"support.citation_count": 2},
    )

    restored = SemanticEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert type(restored.payload) is MessageCompletedPayload

    payload = json.loads(event.model_dump_json())
    payload["kind"] = "message.started"
    with pytest.raises(ValidationError):
        SemanticEvent.model_validate(payload)


def test_semantic_event_round_trips_structured_tool_error_and_turn_outcome() -> None:
    tool_event = SemanticEvent(
        event_id="event_tool_1",
        deployment_id="deployment_1",
        session_id="session_1",
        turn_id="run_1",
        sequence=3,
        producer="agent.support",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        kind=SemanticEventKind.TOOL_ACTIVITY,
        payload=ToolActivityPayload(
            tool_call_id="call_1",
            tool_name="change_agent",
            status="failed",
            message="The Agent changed since it was read.",
            error=ToolErrorPayload(
                kind="rejected",
                code="stale_revision",
                summary="The Agent changed since it was read.",
                retryable=True,
                details={"expected_revision": 4, "actual_revision": 5},
                repair_hints=("Read the current Agent and retry.",),
                allowed_actions=("get_agent",),
                reference="tool-error-1",
            ),
        ),
    )
    restored_tool = SemanticEvent.model_validate_json(tool_event.model_dump_json())
    assert type(restored_tool.payload) is ToolActivityPayload
    assert restored_tool.payload.error is not None
    assert restored_tool.payload.error.code == "stale_revision"

    outcome_event = SemanticEvent(
        event_id="event_outcome_1",
        deployment_id="deployment_1",
        session_id="session_1",
        turn_id="run_1",
        sequence=4,
        producer="agent.support",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        kind=SemanticEventKind.TURN_OUTCOME,
        payload=TurnOutcomePayload(
            outcome="budget_exhausted",
            code="step_budget_exhausted",
            summary="The Agent exhausted its step budget.",
            resumable=False,
            engine_turn_id="engine_turn_1",
            reply_disposition="no_message",
        ),
    )
    restored_outcome = SemanticEvent.model_validate_json(outcome_event.model_dump_json())
    assert type(restored_outcome.payload) is TurnOutcomePayload
    assert restored_outcome.payload.outcome == "budget_exhausted"
    assert restored_outcome.payload.reply_disposition == "no_message"


def test_historical_turn_outcome_preserves_unknown_reply_disposition() -> None:
    payload = TurnOutcomePayload.model_validate(
        {
            "outcome": "completed",
            "code": "completed",
            "summary": "Historical completion.",
            "resumable": False,
            "engine_turn_id": "engine-turn-old",
        }
    )

    assert payload.reply_disposition is None


def test_semantic_event_fixture_freezes_coordinated_manifest_and_event_sequence() -> None:
    fixture = json.loads(_PROTOCOL_FIXTURE_PATH.read_text(encoding="utf-8"))

    manifest = HostManifest.model_validate(fixture["host_manifest"])
    events = tuple(SemanticEvent.model_validate(item) for item in fixture["events"])

    assert manifest.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_VERSION
    assert manifest.release_compatibility.semantic_event_protocol_version == (
        SEMANTIC_EVENT_PROTOCOL_VERSION
    )
    capabilities = manifest.integration_routes[0].required_capabilities
    assert capabilities.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_VERSION
    assert capabilities.event_kinds == (
        SemanticEventKind.TOOL_ACTIVITY,
        SemanticEventKind.WARNING_RAISED,
        SemanticEventKind.TURN_OUTCOME,
    )
    assert tuple(event.sequence for event in events) == (3, 4, 5)
    assert type(events[0].payload) is ToolActivityPayload
    assert events[0].payload.error is not None
    assert events[0].payload.error.code == "stale_revision"
    assert events[1].kind is SemanticEventKind.WARNING_RAISED
    assert type(events[2].payload) is TurnOutcomePayload
    assert events[2].payload.engine_turn_id == "engine_turn_fixture"


def test_semantic_event_rejects_superseded_shape_and_requires_structured_failure() -> None:
    v1_payload = {
        "event_id": "event_1",
        "deployment_id": "deployment_1",
        "session_id": "session_1",
        "turn_id": "run_1",
        "sequence": 1,
        "producer": "agent.support",
        "timestamp": datetime(2026, 8, 3, tzinfo=UTC).isoformat(),
        "kind": "turn.completed",
        "payload": {"result_available": True},
    }
    with pytest.raises(ValidationError):
        SemanticEvent.model_validate(v1_payload)

    failed_activity = ToolActivityPayload(
        tool_call_id="call_1",
        tool_name="change_agent",
        status="failed",
        error=ToolErrorPayload(
            kind="rejected",
            code="stale_revision",
            summary="The Agent changed.",
        ),
    )
    assert failed_activity.error is not None


def test_public_contract_json_schemas_forbid_unknown_object_fields() -> None:
    for contract in (
        HostManifest,
        IntegrationRoute,
        IngressEnvelope,
        SemanticEvent,
    ):
        schema = contract.model_json_schema()
        assert schema["additionalProperties"] is False
