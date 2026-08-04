from datetime import UTC, datetime
import json

from pydantic import ValidationError
import pytest

from aethergraph.contracts.integration import (
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
)
from tests._integration_fixtures import contract_compatibility

_DIGEST = "a" * 64


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "aethergraph.host-manifest/v2"),
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


def test_public_contract_json_schemas_forbid_unknown_object_fields() -> None:
    for contract in (HostManifest, IntegrationRoute, IngressEnvelope, SemanticEvent):
        schema = contract.model_json_schema()
        assert schema["additionalProperties"] is False
