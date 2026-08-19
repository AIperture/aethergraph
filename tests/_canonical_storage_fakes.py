"""Small provider-contract fakes for service/API tests outside conformance."""

from __future__ import annotations

from datetime import UTC, datetime

from aethergraph.services.control.canonical_stores import CanonicalSessionStore
from aethergraph.services.integration import (
    CanonicalInboundEventStore,
    CanonicalIngressIdempotencyStore,
    CanonicalIntegrationPersistence,
    CanonicalIntegrationSessionStore,
    CanonicalSemanticEventStore,
)
from aethergraph.storage.contracts import StorageScope
from tests.storage_conformance.runtime_repositories import (
    InMemoryDeliveryCursorAllocator,
    InMemoryInboundEventRepository,
    InMemoryIngressIdempotencyRepository,
    InMemoryIntegrationSessionRepository,
    InMemorySemanticEventRepository,
    InMemorySessionRepository,
)


def make_session_store() -> CanonicalSessionStore:
    return CanonicalSessionStore(
        repository=InMemorySessionRepository(),
        owner_scope=StorageScope(project_id="test-project"),
        clock=lambda: datetime.now(UTC),
    )


class ClosableCanonicalSemanticEventStore(CanonicalSemanticEventStore):
    async def close(self) -> None:
        """Match test-owned lifecycle without owning a provider bundle."""


def make_semantic_event_store() -> ClosableCanonicalSemanticEventStore:
    return ClosableCanonicalSemanticEventStore(
        repository=InMemorySemanticEventRepository(InMemoryDeliveryCursorAllocator()),
        owner_scope=StorageScope(project_id="test-project"),
    )


def make_integration_persistence() -> CanonicalIntegrationPersistence:
    owner = StorageScope(project_id="test-project")
    sessions = InMemorySessionRepository()
    return CanonicalIntegrationPersistence(
        idempotency=CanonicalIngressIdempotencyStore(
            repository=InMemoryIngressIdempotencyRepository(),
            owner_scope=owner,
            clock=lambda: datetime.now(UTC),
        ),
        sessions=CanonicalIntegrationSessionStore(
            repository=InMemoryIntegrationSessionRepository(sessions),
            owner_scope=owner,
        ),
        inbound_events=CanonicalInboundEventStore(
            repository=InMemoryInboundEventRepository(),
            owner_scope=owner,
        ),
        semantic_events=make_semantic_event_store(),
    )
