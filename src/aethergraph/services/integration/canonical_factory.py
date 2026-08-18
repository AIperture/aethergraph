"""Canonical Host-integration persistence binding over provider storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from aethergraph.storage.contracts import StorageBundle, StorageScope

from .canonical_events import CanonicalInboundEventStore, CanonicalSemanticEventStore
from .event_contracts import InboundEventStore, SemanticEventStore
from .idempotency import CanonicalIngressIdempotencyStore, IngressIdempotencyStore
from .session_bindings import (
    CanonicalExternalSessionBindingStore,
    ExternalSessionBindingStore,
)


@dataclass(frozen=True, slots=True)
class CanonicalIntegrationPersistence:
    """Focused Host integration stores bound to one coherent provider bundle."""

    idempotency: IngressIdempotencyStore
    bindings: ExternalSessionBindingStore
    inbound_events: InboundEventStore
    semantic_events: SemanticEventStore


def bind_canonical_integration_persistence(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalIntegrationPersistence:
    """Bind Host integration persistence to exact canonical bundle fields.

    The binding exposes the existing focused service protocols while claims,
    bindings, accepted ingress, and semantic events flow through their exact bundle
    repositories. It performs no I/O or provider lifecycle work.

    Examples:
        Bind production composition inputs:
            ```python
            persistence = bind_canonical_integration_persistence(
                bundle=bundle,
                owner_scope=open_request.owner_scope,
                clock=open_request.clock.now,
            )
            ```

        Bind a deterministic fake bundle:
            ```python
            persistence = bind_canonical_integration_persistence(
                bundle=fake_bundle,
                owner_scope=StorageScope(project_id="project-1"),
                clock=lambda: fixed_now,
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Exact trusted Host ownership scope from provider composition.
        clock: UTC completion timestamp source shared with runtime composition.

    Returns:
        CanonicalIntegrationPersistence: Focused service-facing store projections.

    Notes:
        This binding does not select or open a provider, resolve a path, activate
        `DefaultContainer`, retry another store, or accept deprecated App metadata.
    """
    return CanonicalIntegrationPersistence(
        idempotency=CanonicalIngressIdempotencyStore(
            repository=bundle.ingress_idempotency,
            owner_scope=owner_scope,
            clock=clock,
        ),
        bindings=CanonicalExternalSessionBindingStore(
            repository=bundle.external_session_bindings,
            owner_scope=owner_scope,
        ),
        inbound_events=CanonicalInboundEventStore(
            repository=bundle.inbound_events,
            owner_scope=owner_scope,
        ),
        semantic_events=CanonicalSemanticEventStore(
            repository=bundle.semantic_events,
            owner_scope=owner_scope,
        ),
    )
