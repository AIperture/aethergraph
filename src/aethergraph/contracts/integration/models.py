"""Closed public contracts for AetherGraph integration and host boundaries."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .versions import (
    EXTERNAL_SESSION_BINDING_SCHEMA_VERSION,
    HOST_MANIFEST_SCHEMA_VERSION,
    INGRESS_ENVELOPE_SCHEMA_VERSION,
    INGRESS_PROTOCOL_VERSION,
    INGRESS_RECEIPT_SCHEMA_VERSION,
    INTEGRATION_CAPABILITIES_SCHEMA_VERSION,
    INTEGRATION_ROUTE_SCHEMA_VERSION,
    ORIGIN_BINDING_SCHEMA_VERSION,
    SEMANTIC_EVENT_PROTOCOL_VERSION,
)

Identifier = Annotated[str, Field(min_length=1, max_length=255)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BoundedText = Annotated[str, Field(max_length=1_000_000)]
MetadataScalar: TypeAlias = str | int | float | bool | None


class IntegrationContract(BaseModel):
    """Base class for immutable and closed integration records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IntegrationKind(StrEnum):
    """Supported transport kinds for the first unified-host protocol."""

    AG_UI = "ag_ui"
    AGENT_ENDPOINT = "agent_endpoint"
    SLACK = "slack"
    TELEGRAM = "telegram"


class SemanticEventKind(StrEnum):
    """Semantic events exposed to transport projectors and endpoint clients."""

    INPUT_ACCEPTED = "input.accepted"
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    PHASE_CHANGED = "phase.changed"
    PROGRESS_CHANGED = "progress.changed"
    INTERACTION_REQUESTED = "interaction.requested"
    INTERACTION_RESOLVED = "interaction.resolved"
    TOOL_ACTIVITY = "tool.activity"
    ARTIFACT_AVAILABLE = "artifact.available"
    STRUCTURED_OUTPUT = "structured.output"
    WARNING_RAISED = "warning.raised"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"


class IntegrationCapabilities(IntegrationContract):
    """Exact capabilities required by a route or offered by an adapter."""

    schema_version: Literal["aethergraph.integration-capabilities/v1"] = (
        INTEGRATION_CAPABILITIES_SCHEMA_VERSION
    )
    semantic_event_protocol_version: Literal["aethergraph.semantic-event/v1"] = (
        SEMANTIC_EVENT_PROTOCOL_VERSION
    )
    event_kinds: tuple[SemanticEventKind, ...]
    streaming: bool
    interactions: bool
    attachments: bool
    cancellation: bool

    @field_validator("event_kinds")
    @classmethod
    def _unique_event_kinds(
        cls, value: tuple[SemanticEventKind, ...]
    ) -> tuple[SemanticEventKind, ...]:
        if len(value) != len(set(value)):
            raise ValueError("event_kinds must not contain duplicates")
        return value


class IntegrationMatchPolicy(IntegrationContract):
    """Authenticated external identities accepted by an integration route."""

    external_tenant_ids: tuple[Identifier, ...] = ()
    external_conversation_ids: tuple[Identifier, ...] = ()
    external_user_ids: tuple[Identifier, ...] = ()


class IntegrationSessionPolicy(IntegrationContract):
    """Explicit external identity scope used to create durable AG sessions."""

    scope: Literal[
        "conversation",
        "conversation_thread",
        "conversation_user",
        "conversation_thread_user",
    ]


class IntegrationRoute(IntegrationContract):
    """Immutable route from one authenticated integration to one entry agent."""

    schema_version: Literal["aethergraph.integration-route/v1"] = INTEGRATION_ROUTE_SCHEMA_VERSION
    route_id: Identifier
    endpoint_id: Identifier | None = None
    integration_id: Identifier
    integration_kind: IntegrationKind
    entry_agent_id: Identifier
    enabled: bool
    match_policy: IntegrationMatchPolicy
    session_policy: IntegrationSessionPolicy
    required_capabilities: IntegrationCapabilities

    @model_validator(mode="after")
    def _validate_endpoint_identity(self) -> IntegrationRoute:
        endpoint_kind = self.integration_kind in {
            IntegrationKind.AG_UI,
            IntegrationKind.AGENT_ENDPOINT,
        }
        if endpoint_kind != (self.endpoint_id is not None):
            raise ValueError("endpoint_id is required only for ag_ui and agent_endpoint routes")
        return self


class HostManifest(IntegrationContract):
    """Immutable launch contract consumed by exactly one AG Host deployment."""

    schema_version: Literal["aethergraph.host-manifest/v1"] = HOST_MANIFEST_SCHEMA_VERSION
    deployment_id: Identifier
    build_id: Identifier
    source_digest: Digest
    build_root: str | None = None
    release_uri: str | None = None
    entrypoint_module: Identifier
    entrypoint_symbol: Identifier
    graph_id: Identifier
    entry_agent_id: Identifier
    environment_snapshot_digest: Digest
    runtime_profile_digest: Digest
    application_settings_digest: Digest
    semantic_event_protocol_version: Literal["aethergraph.semantic-event/v1"] = (
        SEMANTIC_EVENT_PROTOCOL_VERSION
    )
    ingress_protocol_version: Literal["aethergraph.ingress/v1"] = INGRESS_PROTOCOL_VERSION
    integration_routes: tuple[IntegrationRoute, ...]
    logical_output_bindings: dict[Identifier, Identifier] = Field(default_factory=dict)
    workspace_identity: Identifier
    manifest_digest: Digest

    @model_validator(mode="after")
    def _validate_source_and_routes(self) -> HostManifest:
        if (self.build_root is None) == (self.release_uri is None):
            raise ValueError("exactly one of build_root or release_uri is required")
        route_ids = [route.route_id for route in self.integration_routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("integration route_id values must be unique")
        endpoint_ids = [
            route.endpoint_id for route in self.integration_routes if route.endpoint_id is not None
        ]
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("integration endpoint_id values must be unique")
        return self


class ExternalIdentity(IntegrationContract):
    """Authenticated external identity carried by canonical ingress."""

    tenant_id: Identifier
    conversation_id: Identifier
    thread_id: Identifier | None = None
    user_id: Identifier


class OriginAddress(IntegrationContract):
    """Canonical reply address produced only by a verified transport edge."""

    channel_key: Identifier
    capability_profile_id: Identifier


class IngressChoice(IntegrationContract):
    """Exact interaction choice submitted by an external participant."""

    interaction_id: Identifier
    option_ids: tuple[Identifier, ...]

    @field_validator("option_ids")
    @classmethod
    def _require_options(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("option_ids must contain at least one option")
        if len(value) != len(set(value)):
            raise ValueError("option_ids must not contain duplicates")
        return value


class IngressAttachment(IntegrationContract):
    """Provider-neutral reference to one authenticated inbound attachment."""

    attachment_id: Identifier
    source_kind: Literal["provider_file", "artifact"]
    source_id: Identifier
    filename: Annotated[str, Field(min_length=1, max_length=512)]
    content_type: Annotated[str, Field(min_length=1, max_length=255)]
    size_bytes: Annotated[int, Field(ge=0)] | None = None


class IngressEnvelope(IntegrationContract):
    """Canonical command accepted by the unified ingress coordinator."""

    schema_version: Literal["aethergraph.ingress-envelope/v1"] = INGRESS_ENVELOPE_SCHEMA_VERSION
    integration_id: Identifier
    route_hint: Identifier | None = None
    endpoint_id: Identifier | None = None
    external_identity: ExternalIdentity
    external_event_id: Identifier
    idempotency_key: Identifier
    received_at: datetime
    text: BoundedText | None = None
    choice: IngressChoice | None = None
    attachments: tuple[IngressAttachment, ...] = ()
    structured_input: dict[str, JsonValue] | None = None
    transport_metadata: dict[str, MetadataScalar] = Field(default_factory=dict)
    origin_address: OriginAddress

    @model_validator(mode="after")
    def _validate_command(self) -> IngressEnvelope:
        if self.route_hint is not None and self.endpoint_id is not None:
            raise ValueError("route_hint and endpoint_id are mutually exclusive")
        if not any(
            (
                self.text is not None,
                self.choice is not None,
                bool(self.attachments),
                self.structured_input is not None,
            )
        ):
            raise ValueError("ingress must contain text, choice, attachments, or input")
        return self


class IngressReceipt(IntegrationContract):
    """Durable result of accepting or rejecting one canonical ingress event."""

    schema_version: Literal["aethergraph.ingress-receipt/v1"] = INGRESS_RECEIPT_SCHEMA_VERSION
    accepted: bool
    duplicate: bool
    action: Literal["continuation_resumed", "root_turn_started", "rejected"]
    deployment_id: Identifier
    route_id: Identifier | None = None
    session_id: Identifier | None = None
    turn_id: Identifier | None = None
    event_cursor: Annotated[int, Field(ge=0)] | None = None
    rejection_code: Identifier | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> IngressReceipt:
        if self.accepted == (self.action == "rejected"):
            raise ValueError("accepted must be false exactly when action is rejected")
        if self.accepted and (
            self.route_id is None
            or self.session_id is None
            or self.turn_id is None
            or self.event_cursor is None
        ):
            raise ValueError("accepted receipts require route, session, turn, and cursor")
        if self.accepted and self.rejection_code is not None:
            raise ValueError("accepted receipts cannot include rejection_code")
        if not self.accepted and self.rejection_code is None:
            raise ValueError("rejected receipts require rejection_code")
        return self


class ExternalSessionBinding(IntegrationContract):
    """Durable association between an external conversation and an AG session."""

    schema_version: Literal["aethergraph.external-session-binding/v1"] = (
        EXTERNAL_SESSION_BINDING_SCHEMA_VERSION
    )
    binding_id: Identifier
    route_id: Identifier
    external_identity: ExternalIdentity
    ag_session_id: Identifier
    build_id: Identifier
    created_at: datetime
    last_seen_at: datetime

    @model_validator(mode="after")
    def _validate_timestamps(self) -> ExternalSessionBinding:
        if self.last_seen_at < self.created_at:
            raise ValueError("last_seen_at cannot precede created_at")
        return self


class OriginBinding(IntegrationContract):
    """Immutable run-scoped origin used by the default Channel session."""

    schema_version: Literal["aethergraph.origin-binding/v1"] = ORIGIN_BINDING_SCHEMA_VERSION
    integration_id: Identifier
    route_id: Identifier
    session_id: Identifier
    channel_key: Identifier
    external_conversation_id: Identifier
    external_thread_id: Identifier | None = None
    capability_profile_id: Identifier


class InputAcceptedPayload(IntegrationContract):
    """Semantic payload recording accepted external user input."""

    input_id: Identifier
    text: BoundedText | None = None
    artifacts: tuple[ArtifactAvailablePayload, ...] = ()
    interaction_id: Identifier | None = None
    option_ids: tuple[Identifier, ...] = ()


class MessageStartedPayload(IntegrationContract):
    """Semantic payload opening one streamed or buffered assistant message."""

    message_id: Identifier


class MessageDeltaPayload(IntegrationContract):
    """Semantic payload adding ordered text to an open message."""

    message_id: Identifier
    delta: BoundedText


class MessageCompletedPayload(IntegrationContract):
    """Semantic payload completing one assistant message."""

    message_id: Identifier
    text: BoundedText
    artifact_ids: tuple[Identifier, ...] = ()


class PhaseChangedPayload(IntegrationContract):
    """Semantic payload describing a named execution phase transition."""

    phase: Identifier
    status: Literal["pending", "active", "done", "failed", "skipped"]
    label: Annotated[str, Field(min_length=1, max_length=255)]
    detail: Annotated[str, Field(max_length=4_000)] | None = None


class ProgressChangedPayload(IntegrationContract):
    """Semantic payload reporting bounded or indeterminate progress."""

    progress_id: Identifier
    status: Literal["started", "running", "completed", "failed", "canceled"]
    label: Annotated[str, Field(min_length=1, max_length=255)]
    current: float | None = None
    total: float | None = None
    unit: Annotated[str, Field(max_length=64)] | None = None
    detail: Annotated[str, Field(max_length=4_000)] | None = None


class InteractionOption(IntegrationContract):
    """Authored option exposed by a semantic interaction request."""

    option_id: Identifier
    label: Annotated[str, Field(min_length=1, max_length=255)]


class InteractionRequestedPayload(IntegrationContract):
    """Semantic payload requesting an exact external interaction."""

    interaction_id: Identifier
    request_kind: Literal["approval", "choice", "text", "files"]
    prompt: BoundedText
    options: tuple[InteractionOption, ...] = ()
    allow_multiple: bool = False
    accepted_content_types: tuple[Annotated[str, Field(max_length=255)], ...] = ()


class InteractionResolvedPayload(IntegrationContract):
    """Semantic payload recording the exact result of an interaction."""

    interaction_id: Identifier
    resolution_kind: Literal["approved", "rejected", "choice", "text", "files"]
    option_ids: tuple[Identifier, ...] = ()
    text: BoundedText | None = None
    artifact_ids: tuple[Identifier, ...] = ()


class ToolActivityPayload(IntegrationContract):
    """Semantic payload reporting transport-neutral Tool activity."""

    tool_call_id: Identifier
    tool_name: Identifier
    status: Literal["started", "running", "waiting", "completed", "failed", "canceled"]
    message: Annotated[str, Field(max_length=4_000)] | None = None


class ArtifactAvailablePayload(IntegrationContract):
    """Semantic payload exposing one downloadable runtime artifact."""

    artifact_id: Identifier
    filename: Annotated[str, Field(min_length=1, max_length=512)]
    content_type: Annotated[str, Field(min_length=1, max_length=255)]
    size_bytes: Annotated[int, Field(ge=0)]


class StructuredOutputPayload(IntegrationContract):
    """Semantic payload carrying one explicitly named JSON output."""

    output_name: Identifier
    value: JsonValue


class WarningRaisedPayload(IntegrationContract):
    """Semantic payload reporting a non-terminal authored warning."""

    code: Identifier
    message: Annotated[str, Field(min_length=1, max_length=4_000)]
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TurnCompletedPayload(IntegrationContract):
    """Semantic payload marking successful terminal turn completion."""

    result_available: bool


class TurnFailedPayload(IntegrationContract):
    """Semantic payload marking terminal turn failure."""

    code: Identifier
    message: Annotated[str, Field(min_length=1, max_length=4_000)]
    retryable: bool


SemanticPayload: TypeAlias = (
    InputAcceptedPayload
    | MessageStartedPayload
    | MessageDeltaPayload
    | MessageCompletedPayload
    | PhaseChangedPayload
    | ProgressChangedPayload
    | InteractionRequestedPayload
    | InteractionResolvedPayload
    | ToolActivityPayload
    | ArtifactAvailablePayload
    | StructuredOutputPayload
    | WarningRaisedPayload
    | TurnCompletedPayload
    | TurnFailedPayload
)


_PAYLOAD_BY_KIND: dict[SemanticEventKind, type[IntegrationContract]] = {
    SemanticEventKind.INPUT_ACCEPTED: InputAcceptedPayload,
    SemanticEventKind.MESSAGE_STARTED: MessageStartedPayload,
    SemanticEventKind.MESSAGE_DELTA: MessageDeltaPayload,
    SemanticEventKind.MESSAGE_COMPLETED: MessageCompletedPayload,
    SemanticEventKind.PHASE_CHANGED: PhaseChangedPayload,
    SemanticEventKind.PROGRESS_CHANGED: ProgressChangedPayload,
    SemanticEventKind.INTERACTION_REQUESTED: InteractionRequestedPayload,
    SemanticEventKind.INTERACTION_RESOLVED: InteractionResolvedPayload,
    SemanticEventKind.TOOL_ACTIVITY: ToolActivityPayload,
    SemanticEventKind.ARTIFACT_AVAILABLE: ArtifactAvailablePayload,
    SemanticEventKind.STRUCTURED_OUTPUT: StructuredOutputPayload,
    SemanticEventKind.WARNING_RAISED: WarningRaisedPayload,
    SemanticEventKind.TURN_COMPLETED: TurnCompletedPayload,
    SemanticEventKind.TURN_FAILED: TurnFailedPayload,
}


class SemanticEvent(IntegrationContract):
    """Ordered transport-neutral event emitted by an AG execution turn."""

    schema_version: Literal["aethergraph.semantic-event/v1"] = SEMANTIC_EVENT_PROTOCOL_VERSION
    event_id: Identifier
    deployment_id: Identifier
    session_id: Identifier
    turn_id: Identifier
    sequence: Annotated[int, Field(ge=0)]
    producer: Identifier
    timestamp: datetime
    kind: SemanticEventKind
    payload: SemanticPayload
    extensions: dict[Identifier, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_payload_for_kind(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw_kind = value.get("kind")
        try:
            kind = SemanticEventKind(raw_kind)
        except (TypeError, ValueError):
            return value
        raw_payload = value.get("payload")
        if isinstance(raw_payload, dict):
            parsed = dict(value)
            parsed["payload"] = _PAYLOAD_BY_KIND[kind].model_validate(raw_payload)
            return parsed
        return value

    @model_validator(mode="after")
    def _validate_payload_kind(self) -> SemanticEvent:
        expected_type = _PAYLOAD_BY_KIND[self.kind]
        if type(self.payload) is not expected_type:
            raise ValueError(f"payload for {self.kind.value} must be {expected_type.__name__}")
        for key in self.extensions:
            if "." not in key:
                raise ValueError("semantic extension keys must be namespaced")
        return self
