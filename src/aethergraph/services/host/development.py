"""Development-only AG UI endpoint composition."""

from __future__ import annotations

from hashlib import sha256
import platform
import sys

from aethergraph import __version__ as AETHERGRAPH_VERSION
from aethergraph.contracts.integration import (
    HostManifest,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    ReleaseCompatibility,
    ReleaseDependency,
    SemanticEventKind,
)

from .manifest import seal_host_manifest

_ZERO_DIGEST = "0" * 64


def build_development_ui_manifest(*, registry, workspace_identity: str) -> HostManifest:
    """Build an immutable endpoint manifest for the mutable development sidecar.

    The development server creates one AG UI endpoint per registered agent so the
    browser uses the same canonical ingress and semantic-event pipeline as an
    immutable deployment. The resulting manifest is process-local and is never a
    compiled-release attestation.

    Examples:
        Build routes for the built-in Chat agent:
            ```python
            manifest = build_development_ui_manifest(
                registry=container.registry,
                workspace_identity="C:/workspace",
            )
            assert manifest.integration_routes
            ```

        Inspect the selected endpoint agent:
            ```python
            route = manifest.integration_routes[0]
            assert route.integration_kind is IntegrationKind.AG_UI
            ```

    Args:
        registry: Unified Registry containing the currently registered agents.
        workspace_identity: Stable development workspace identity for this server.

    Returns:
        HostManifest: Sealed process-local manifest with one route per agent.

    Notes:
        Development manifests deliberately describe mutable local registrations;
        immutable AG Host construction continues to use compiled manifests only.
    """

    agent_entries = registry.list_agents(include_global=True)
    agents: list[tuple[str, str]] = []
    for reference in sorted(agent_entries):
        _, name = reference.split(":", 1)
        meta = registry.get_meta(nspace="agent", name=name, include_global=True) or {}
        agent_id = str(meta.get("id") or name)
        agents.append((agent_id, name))
    if not agents:
        raise RuntimeError("Development AG UI requires at least one registered agent.")

    event_kinds = tuple(SemanticEventKind)
    routes = tuple(
        IntegrationRoute(
            route_id=f"development-ui-route-{_short_digest(agent_id)}",
            endpoint_id=f"development-ui-{_short_digest(agent_id)}",
            integration_id=f"development-ui-{_short_digest(agent_id)}",
            integration_kind=IntegrationKind.AG_UI,
            entry_agent_id=agent_id,
            enabled=True,
            match_policy=IntegrationMatchPolicy(),
            session_policy=IntegrationSessionPolicy(scope="conversation_user"),
            required_capabilities=IntegrationCapabilities(
                event_kinds=event_kinds,
                streaming=True,
                interactions=True,
                attachments=True,
                cancellation=True,
            ),
        )
        for agent_id, _ in agents
    )
    agent_digest = sha256("\n".join(agent_id for agent_id, _ in agents).encode()).hexdigest()
    compatibility = ReleaseCompatibility(
        aethergraph_version=AETHERGRAPH_VERSION,
        engine_version="development",
        python_abi=sys.implementation.cache_tag or "unknown",
        platform=sys.platform,
        architecture=platform.machine() or platform.architecture()[0],
        dependency_lock=(
            ReleaseDependency(
                name="aethergraph",
                version=AETHERGRAPH_VERSION,
                content_sha256=agent_digest,
            ),
        ),
        dependency_lock_digest=agent_digest,
        host_capability_requirements=("canonical_ingress",),
        service_requirements=("channels", "eventlog", "runs", "sessions"),
        logical_output_requirements=("origin",),
        entrypoint_input_schema={"type": "object"},
        entrypoint_output_schema={"type": "object"},
        compiled_manifest_sha256=agent_digest,
        provenance={"mode": "development"},
    )
    first_agent_id, first_graph_name = agents[0]
    manifest = HostManifest(
        deployment_id=f"development-{_short_digest(workspace_identity)}",
        build_id=f"development-{agent_digest[:16]}",
        source_digest=agent_digest,
        build_root=workspace_identity,
        entrypoint_module="aethergraph.development",
        entrypoint_symbol="register",
        graph_id=first_graph_name,
        entry_agent_id=first_agent_id,
        environment_snapshot_digest=_ZERO_DIGEST,
        runtime_profile_digest=_ZERO_DIGEST,
        application_settings_digest=_ZERO_DIGEST,
        release_compatibility=compatibility,
        integration_routes=routes,
        workspace_identity=workspace_identity,
        manifest_digest=_ZERO_DIGEST,
    )
    return seal_host_manifest(manifest)


def development_ui_endpoints(manifest: HostManifest) -> tuple[dict[str, str], ...]:
    """Project development AG UI routes into a browser-safe endpoint catalog.

    The projection exposes only agent and endpoint identities. It does not expose
    release metadata, workspace paths, or endpoint credentials.

    Examples:
        Project a development manifest:
            ```python
            endpoints = development_ui_endpoints(manifest)
            assert endpoints[0]["agent_id"]
            ```

        Read an endpoint identity:
            ```python
            endpoint_id = development_ui_endpoints(manifest)[0]["endpoint_id"]
            ```

    Args:
        manifest: Development manifest whose enabled AG UI routes are projected.

    Returns:
        tuple[dict[str, str], ...]: Ordered agent-to-endpoint mappings.

    Notes:
        Disabled and non-AG-UI routes are excluded.
    """

    return tuple(
        {"agent_id": route.entry_agent_id, "endpoint_id": str(route.endpoint_id)}
        for route in manifest.integration_routes
        if route.enabled and route.integration_kind is IntegrationKind.AG_UI
    )


def _short_digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()[:20]
