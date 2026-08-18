"""Explicit AG Host installation for the canonical integration coordinator."""

from __future__ import annotations

from aethergraph.contracts.integration import (
    SEMANTIC_EVENT_PROTOCOL_VERSION,
    HostManifest,
    IntegrationKind,
)

from .coordinator import IntegrationIngressCoordinator
from .delivery import SemanticEventChannelAdapter, SemanticEventEmitter, SemanticTurnMonitor
from .dispatch import AGRootTurnDispatcher
from .interactions import InteractionResolver
from .resources import ResourceIngress, ResourceIngressPolicy
from .routes import ManifestRouteResolver


def install_integration_ingress(
    *,
    container,
    manifest: HostManifest,
    resource_policy: ResourceIngressPolicy | None = None,
) -> IntegrationIngressCoordinator:
    """Install the one manifest-bound ingress coordinator on an AG Host.

    The installer composes the provider-neutral stores and execution services
    from one container and one immutable deployment manifest.

    Examples:
        Install using the selected canonical provider:
        ```python
        coordinator = install_integration_ingress(
            container=container,
            manifest=manifest,
        )
        ```

        Install with an explicit resource policy:
        ```python
        coordinator = install_integration_ingress(
            container=container,
            manifest=manifest,
            resource_policy=policy,
        )
        ```

    Args:
        container: Fully built AG Host service container.
        manifest: Immutable deployment manifest and sole route authority.
        resource_policy: Optional shared attachment validation policy.

    Returns:
        IntegrationIngressCoordinator: Installed canonical ingress boundary.

    Notes:
        Reinstalling over an active coordinator is rejected. A manifest revision
        requires a new Host process, so route authority cannot mutate in place.
    """
    if getattr(container, "integration_ingress", None) is not None:
        raise RuntimeError("Integration ingress is already installed on this Host.")
    if manifest.semantic_event_protocol_version != SEMANTIC_EVENT_PROTOCOL_VERSION:
        raise RuntimeError(
            "Semantic event protocol "
            f"{manifest.semantic_event_protocol_version!r} is negotiated but its "
            "delivery projector is not enabled."
        )
    persistence = container.storage_services.integration
    semantic_events = persistence.semantic_events
    emitter = SemanticEventEmitter(
        deployment_id=manifest.deployment_id,
        store=semantic_events,
        semantic_event_protocol_version=manifest.semantic_event_protocol_version,
    )
    turn_monitor = SemanticTurnMonitor(
        run_manager=container.run_manager,
        emitter=emitter,
    )
    container.channels.register_adapter(
        "endpoint",
        SemanticEventChannelAdapter(emitter=emitter),
    )
    provider_prefixes = {
        IntegrationKind.SLACK: "slack",
        IntegrationKind.TELEGRAM: "tg",
    }
    enabled_kinds = {
        route.integration_kind for route in manifest.integration_routes if route.enabled
    }
    for kind, prefix in provider_prefixes.items():
        if kind not in enabled_kinds:
            continue
        downstream = container.channels.adapters.get(prefix)
        if downstream is None:
            raise RuntimeError(f"Enabled {kind.value} route requires the {prefix!r} adapter.")
        container.channels.register_adapter(
            prefix,
            SemanticEventChannelAdapter(emitter=emitter, downstream=downstream),
        )

    coordinator = IntegrationIngressCoordinator(
        manifest=manifest,
        route_resolver=ManifestRouteResolver(manifest),
        idempotency_store=persistence.idempotency,
        binding_store=persistence.bindings,
        resource_ingress=ResourceIngress(container=container, policy=resource_policy),
        interaction_resolver=InteractionResolver(container.cont_store),
        inbound_events=persistence.inbound_events,
        semantic_emitter=emitter,
        resume_router=container.resume_router,
        root_dispatcher=AGRootTurnDispatcher(
            container,
            turn_monitor=turn_monitor,
        ),
    )
    container.host_manifest = manifest
    container.semantic_events = semantic_events
    container.semantic_turn_monitor = turn_monitor
    container.integration_ingress = coordinator
    return coordinator
