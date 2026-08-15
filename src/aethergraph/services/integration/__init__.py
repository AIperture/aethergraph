"""Provider-neutral services for unified integration ingress."""

from .canonical_events import CanonicalInboundEventStore, CanonicalSemanticEventStore
from .canonical_factory import (
    CanonicalIntegrationPersistence,
    bind_canonical_integration_persistence,
)
from .context import VerifiedAttachment, VerifiedIntegrationContext
from .coordinator import IngressCoordinatorError, IntegrationIngressCoordinator
from .delivery import (
    SemanticDeliveryError,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
    SemanticTurnMonitor,
)
from .dispatch import AGRootTurnDispatcher, RootTurnDispatcher
from .events import (
    EventLogInboundEventStore,
    EventLogSemanticEventStore,
    InboundEventStore,
    PersistedInboundEvent,
    PersistedSemanticEvent,
    SemanticEventStore,
    SemanticEventStoreError,
)
from .factory import install_integration_ingress
from .idempotency import (
    CanonicalIngressIdempotencyStore,
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
    CanonicalExternalSessionBindingStore,
    ExternalSessionBindingStore,
    SessionBindingError,
    SQLiteExternalSessionBindingStore,
)

__all__ = [
    "BindingResolution",
    "CanonicalExternalSessionBindingStore",
    "CanonicalIngressIdempotencyStore",
    "CanonicalInboundEventStore",
    "CanonicalIntegrationPersistence",
    "CanonicalSemanticEventStore",
    "AGRootTurnDispatcher",
    "EventLogInboundEventStore",
    "ExternalSessionBindingStore",
    "EventLogSemanticEventStore",
    "IngressClaim",
    "IngressCoordinatorError",
    "IngressIdempotencyError",
    "IngressIdempotencyStore",
    "InboundEventStore",
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
    "SemanticTurnMonitor",
    "VerifiedIntegrationContext",
    "VerifiedAttachment",
    "build_interaction_payload",
    "bind_canonical_integration_persistence",
    "install_integration_ingress",
]
