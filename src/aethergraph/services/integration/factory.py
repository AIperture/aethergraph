"""Explicit AG Host installation for the canonical integration coordinator."""

from __future__ import annotations

from pathlib import Path

from aethergraph.contracts.integration import HostManifest

from .coordinator import IntegrationIngressCoordinator
from .dispatch import AGRootTurnDispatcher
from .events import EventLogInboundEventStore
from .idempotency import SQLiteIngressIdempotencyStore
from .interactions import InteractionResolver
from .resources import ResourceIngress, ResourceIngressPolicy
from .routes import ManifestRouteResolver
from .session_bindings import SQLiteExternalSessionBindingStore


def install_integration_ingress(
    *,
    container,
    manifest: HostManifest,
    database_path: str | Path | None = None,
    resource_policy: ResourceIngressPolicy | None = None,
) -> IntegrationIngressCoordinator:
    """Install the one manifest-bound ingress coordinator on an AG Host.

    The installer composes the provider-neutral stores and execution services
    from one container and one immutable deployment manifest.

    Examples:
        Install using the Host workspace database:
        ```python
        coordinator = install_integration_ingress(
            container=container,
            manifest=manifest,
        )
        ```

        Install with an explicit operational database:
        ```python
        coordinator = install_integration_ingress(
            container=container,
            manifest=manifest,
            database_path="host/integration/operations.db",
        )
        ```

    Args:
        container: Fully built AG Host service container.
        manifest: Immutable deployment manifest and sole route authority.
        database_path: Optional explicit SQLite operational-store path.
        resource_policy: Optional shared attachment validation policy.

    Returns:
        IntegrationIngressCoordinator: Installed canonical ingress boundary.

    Notes:
        Reinstalling over an active coordinator is rejected. A manifest revision
        requires a new Host process, so route authority cannot mutate in place.
    """
    if getattr(container, "integration_ingress", None) is not None:
        raise RuntimeError("Integration ingress is already installed on this Host.")
    path = (
        Path(database_path)
        if database_path is not None
        else Path(container.root) / ("integration/operations.db")
    )
    coordinator = IntegrationIngressCoordinator(
        manifest=manifest,
        route_resolver=ManifestRouteResolver(manifest),
        idempotency_store=SQLiteIngressIdempotencyStore(path),
        binding_store=SQLiteExternalSessionBindingStore(path),
        resource_ingress=ResourceIngress(container=container, policy=resource_policy),
        interaction_resolver=InteractionResolver(container.cont_store),
        inbound_events=EventLogInboundEventStore(container.eventlog),
        resume_router=container.resume_router,
        root_dispatcher=AGRootTurnDispatcher(container),
    )
    container.host_manifest = manifest
    container.integration_ingress = coordinator
    return coordinator
