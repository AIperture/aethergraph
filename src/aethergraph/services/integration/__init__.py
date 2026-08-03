"""Provider-neutral services for unified integration ingress."""

from .context import VerifiedAttachment, VerifiedIntegrationContext
from .coordinator import IngressCoordinatorError, IntegrationIngressCoordinator
from .delivery import (
    SemanticDeliveryError,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
)
from .dispatch import AGRootTurnDispatcher, RootTurnDispatcher
from .events import (
    EventLogInboundEventStore,
    EventLogSemanticEventStore,
    PersistedInboundEvent,
    PersistedSemanticEvent,
    SemanticEventStore,
    SemanticEventStoreError,
)
from .factory import install_integration_ingress
from .idempotency import (
    IngressClaim,
    IngressIdempotencyError,
    IngressIdempotencyStore,
    SQLiteIngressIdempotencyStore,
)
from .interactions import (
    InteractionResolutionError,
    InteractionResolver,
    ResolvedInteraction,
    build_interaction_payload,
)
from .manager import (
    IntegrationConnection,
    IntegrationConnectionState,
    IntegrationConnectionStatus,
    IntegrationManager,
    IntegrationManagerError,
    IntegrationTransport,
)
from .resources import ResourceIngress, ResourceIngressError, ResourceIngressPolicy
from .routes import IntegrationRouteError, ManifestRouteResolver
from .session_bindings import (
    BindingResolution,
    ExternalSessionBindingStore,
    SessionBindingError,
    SQLiteExternalSessionBindingStore,
)

__all__ = [
    "BindingResolution",
    "AGRootTurnDispatcher",
    "EventLogInboundEventStore",
    "ExternalSessionBindingStore",
    "EventLogSemanticEventStore",
    "IngressClaim",
    "IngressCoordinatorError",
    "IngressIdempotencyError",
    "IngressIdempotencyStore",
    "IntegrationRouteError",
    "IntegrationIngressCoordinator",
    "InteractionResolutionError",
    "InteractionResolver",
    "IntegrationConnection",
    "IntegrationConnectionState",
    "IntegrationConnectionStatus",
    "IntegrationManager",
    "IntegrationManagerError",
    "IntegrationTransport",
    "ManifestRouteResolver",
    "PersistedSemanticEvent",
    "PersistedInboundEvent",
    "ResolvedInteraction",
    "ResourceIngress",
    "ResourceIngressError",
    "ResourceIngressPolicy",
    "RootTurnDispatcher",
    "SQLiteExternalSessionBindingStore",
    "SQLiteIngressIdempotencyStore",
    "SessionBindingError",
    "SemanticEventStore",
    "SemanticEventStoreError",
    "SemanticDeliveryError",
    "SemanticEventChannelAdapter",
    "SemanticEventEmitter",
    "VerifiedIntegrationContext",
    "VerifiedAttachment",
    "build_interaction_payload",
    "install_integration_ingress",
]
