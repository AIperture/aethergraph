"""Protocol identities for transport-neutral AetherGraph integrations."""

from typing import Final

INGRESS_PROTOCOL_VERSION: Final = "aethergraph.ingress/v2"
SEMANTIC_EVENT_PROTOCOL_VERSION: Final = "aethergraph.semantic-event/v3"
SEMANTIC_EVENT_CODEC_REVISION: Final = "aethergraph.semantic-event-codec/v1"
SEMANTIC_EVENT_READ_VERSIONS: Final = (
    "aethergraph.semantic-event/v1",
    "aethergraph.semantic-event/v2",
    SEMANTIC_EVENT_PROTOCOL_VERSION,
)
HOST_MANIFEST_SCHEMA_VERSION: Final = "aethergraph.host-manifest/v4"
HOST_READY_PROTOCOL_VERSION: Final = "aethergraph.host-ready/v2"
HOST_DIAGNOSTIC_SCHEMA_VERSION: Final = "aethergraph.host-diagnostic/v2"
INTEGRATION_ROUTE_SCHEMA_VERSION: Final = "aethergraph.integration-route/v2"
INGRESS_ENVELOPE_SCHEMA_VERSION: Final = "aethergraph.ingress-envelope/v2"
INGRESS_RECEIPT_SCHEMA_VERSION: Final = "aethergraph.ingress-receipt/v1"
EXTERNAL_SESSION_BINDING_SCHEMA_VERSION: Final = "aethergraph.external-session-binding/v1"
ORIGIN_BINDING_SCHEMA_VERSION: Final = "aethergraph.origin-binding/v1"
INTEGRATION_CAPABILITIES_SCHEMA_VERSION: Final = "aethergraph.integration-capabilities/v2"
RELEASE_COMPATIBILITY_SCHEMA_VERSION: Final = "aethergraph.release-compatibility/v3"
