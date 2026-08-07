from datetime import UTC, datetime
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from aethergraph.contracts.integration import (
    INTEGRATION_CAPABILITIES_SCHEMA_VERSION,
    INTEGRATION_ROUTE_SCHEMA_VERSION,
    RELEASE_COMPATIBILITY_SCHEMA_VERSION,
    SEMANTIC_EVENT_PROTOCOL_V2,
    SEMANTIC_EVENT_PROTOCOL_VERSION,
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
    SemanticEventKindV2,
    SemanticEventV2,
    ToolActivityPayload,
    ToolActivityPayloadV2,
    ToolErrorPayload,
    TurnOutcomePayload,
)
from tests._integration_fixtures import contract_compatibility

_DIGEST = "a" * 64
_PROTOCOL_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "integration" / "semantic_event_v2.json"
)


def _capabilities() -> IntegrationCapabilities:
    return IntegrationCapabilities(
        event_kinds=(
            SemanticEventKind.MESSAGE_STARTED,
            SemanticEventKind.MESSAGE_DELTA,
            SemanticEventKind.MESSAGE_COMPLETED,
            SemanticEventKind.TURN_COMPLETED,
            SemanticEventKind.TURN_FAILED,
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

    assert manifest.schema_version == "aethergraph.host-manifest/v3"
    assert manifest.release_compatibility.schema_version == (RELEASE_COMPATIBILITY_SCHEMA_VERSION)
    assert manifest.integration_routes[0].schema_version == INTEGRATION_ROUTE_SCHEMA_VERSION
    assert manifest.integration_routes[0].required_capabilities.schema_version == (
        INTEGRATION_CAPABILITIES_SCHEMA_VERSION
    )
    assert SEMANTIC_EVENT_PROTOCOL_VERSION == "aethergraph.semantic-event/v1"
    assert SEMANTIC_EVENT_PROTOCOL_V2 == "aethergraph.semantic-event/v2"


def test_manifest_accepts_only_a_coordinated_v2_semantic_protocol_selection() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["semantic_event_protocol_version"] = SEMANTIC_EVENT_PROTOCOL_V2
    payload["release_compatibility"]["semantic_event_protocol_version"] = SEMANTIC_EVENT_PROTOCOL_V2
    route_capabilities = payload["integration_routes"][0]["required_capabilities"]
    route_capabilities["semantic_event_protocol_version"] = SEMANTIC_EVENT_PROTOCOL_V2
    route_capabilities["event_kinds"] = [
        "message.started",
        "message.delta",
        "message.completed",
        "turn.outcome",
    ]

    manifest = HostManifest.model_validate(payload)

    assert manifest.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_V2
    assert manifest.integration_routes[0].required_capabilities.event_kinds[-1] == (
        SemanticEventKindV2.TURN_OUTCOME
    )
    assert all(
        isinstance(kind, SemanticEventKindV2)
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


@pytest.mark.parametrize(
    ("protocol_version", "event_kind"),
    [
        (SEMANTIC_EVENT_PROTOCOL_VERSION, SemanticEventKindV2.TURN_OUTCOME),
        (SEMANTIC_EVENT_PROTOCOL_V2, SemanticEventKind.TURN_COMPLETED),
    ],
)
def test_capabilities_reject_event_kinds_from_another_protocol(
    protocol_version: str,
    event_kind: SemanticEventKind | SemanticEventKindV2,
) -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        IntegrationCapabilities(
            semantic_event_protocol_version=protocol_version,
            event_kinds=(event_kind,),
            streaming=False,
            interactions=False,
            attachments=False,
            cancellation=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "aethergraph.host-manifest/v1"),
        ("ingress_protocol_version", "aethergraph.ingress/v2"),
        ("semantic_event_protocol_version", "aethergraph.semantic-event/v2"),
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
        text="Hello",
        origin_address=OriginAddress(
            channel_key="endpoint:sessions/session_1",
            capability_profile_id="ag_ui_v1",
        ),
    )

    restored = IngressEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope

    payload = envelope.model_dump(mode="json")
    payload["text"] = None
    with pytest.raises(ValidationError, match="ingress must contain"):
        IngressEnvelope.model_validate(payload)

    payload["text"] = "Hello"
    payload["raw_provider_payload"] = {"unbounded": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IngressEnvelope.model_validate(payload)


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


def test_semantic_event_v2_round_trips_structured_tool_error_and_turn_outcome() -> None:
    tool_event = SemanticEventV2(
        event_id="event_tool_1",
        deployment_id="deployment_1",
        session_id="session_1",
        turn_id="run_1",
        sequence=3,
        producer="agent.support",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        kind=SemanticEventKindV2.TOOL_ACTIVITY,
        payload=ToolActivityPayloadV2(
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
    restored_tool = SemanticEventV2.model_validate_json(tool_event.model_dump_json())
    assert type(restored_tool.payload) is ToolActivityPayloadV2
    assert restored_tool.payload.error is not None
    assert restored_tool.payload.error.code == "stale_revision"

    outcome_event = SemanticEventV2(
        event_id="event_outcome_1",
        deployment_id="deployment_1",
        session_id="session_1",
        turn_id="run_1",
        sequence=4,
        producer="agent.support",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        kind=SemanticEventKindV2.TURN_OUTCOME,
        payload=TurnOutcomePayload(
            outcome="budget_exhausted",
            code="step_budget_exhausted",
            summary="The Agent exhausted its step budget.",
            resumable=False,
            engine_turn_id="engine_turn_1",
        ),
    )
    restored_outcome = SemanticEventV2.model_validate_json(outcome_event.model_dump_json())
    assert type(restored_outcome.payload) is TurnOutcomePayload
    assert restored_outcome.payload.outcome == "budget_exhausted"


def test_semantic_event_v2_fixture_freezes_coordinated_manifest_and_event_sequence() -> None:
    fixture = json.loads(_PROTOCOL_FIXTURE_PATH.read_text(encoding="utf-8"))

    manifest = HostManifest.model_validate(fixture["host_manifest"])
    events = tuple(SemanticEventV2.model_validate(item) for item in fixture["events"])

    assert manifest.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_V2
    assert manifest.release_compatibility.semantic_event_protocol_version == (
        SEMANTIC_EVENT_PROTOCOL_V2
    )
    capabilities = manifest.integration_routes[0].required_capabilities
    assert capabilities.semantic_event_protocol_version == SEMANTIC_EVENT_PROTOCOL_V2
    assert capabilities.event_kinds == (
        SemanticEventKindV2.TOOL_ACTIVITY,
        SemanticEventKindV2.WARNING_RAISED,
        SemanticEventKindV2.TURN_OUTCOME,
    )
    assert tuple(event.sequence for event in events) == (3, 4, 5)
    assert type(events[0].payload) is ToolActivityPayloadV2
    assert events[0].payload.error is not None
    assert events[0].payload.error.code == "stale_revision"
    assert events[1].kind is SemanticEventKindV2.WARNING_RAISED
    assert type(events[2].payload) is TurnOutcomePayload
    assert events[2].payload.engine_turn_id == "engine_turn_fixture"
    with pytest.raises(ValidationError):
        SemanticEvent.model_validate(fixture["events"][0])


def test_semantic_event_versions_reject_shape_and_terminal_kind_mixing() -> None:
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
        SemanticEventV2.model_validate(v1_payload)

    v2_payload = {
        **v1_payload,
        "schema_version": "aethergraph.semantic-event/v2",
        "kind": "turn.outcome",
        "payload": {
            "outcome": "completed",
            "code": "completed",
            "summary": "Completed.",
            "resumable": False,
            "engine_turn_id": "engine_turn_1",
        },
    }
    with pytest.raises(ValidationError):
        SemanticEvent.model_validate(v2_payload)

    failed_activity = ToolActivityPayloadV2(
        tool_call_id="call_1",
        tool_name="change_agent",
        status="failed",
        error=ToolErrorPayload(
            kind="rejected",
            code="stale_revision",
            summary="The Agent changed.",
        ),
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ToolActivityPayload.model_validate(failed_activity)


def test_public_contract_json_schemas_forbid_unknown_object_fields() -> None:
    for contract in (
        HostManifest,
        IntegrationRoute,
        IngressEnvelope,
        SemanticEvent,
        SemanticEventV2,
    ):
        schema = contract.model_json_schema()
        assert schema["additionalProperties"] is False
