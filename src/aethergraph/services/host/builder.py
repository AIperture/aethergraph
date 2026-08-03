"""Authoritative composition of one immutable local AG Host runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from aethergraph.config.config import AppSettings
from aethergraph.contracts.integration import HostManifest, IntegrationKind
from aethergraph.core.runtime.runtime_registry import set_current_registry
from aethergraph.core.runtime.runtime_services import install_services, use_services
from aethergraph.services.container.default_container import (
    DefaultContainer,
    build_default_container,
)
from aethergraph.services.integration import (
    IntegrationConnection,
    IntegrationManager,
    IntegrationTransport,
    install_integration_ingress,
)
from aethergraph.services.registry.unified_registry import UnifiedRegistry

from .manifest import (
    HostManifestError,
    compute_host_manifest_digest,
    validate_compiled_build,
)


@dataclass(frozen=True)
class HostRuntimeIdentity:
    """Supervisor-verified runtime snapshots consumed by AG Host."""

    environment_snapshot_digest: str
    runtime_profile_digest: str
    application_settings_digest: str


@dataclass(frozen=True)
class HostProviderConnection:
    """Explicit provider delivery and transport construction selected at launch."""

    integration_id: str
    integration_kind: IntegrationKind
    delivery_adapter: object
    transport_factory: Callable[[DefaultContainer], IntegrationTransport]
    close_delivery: Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class AGHost:
    """Fully composed immutable AG Host awaiting application startup."""

    manifest: HostManifest
    runtime_identity: HostRuntimeIdentity
    workspace: Path
    inspection: Any
    entrypoint: Any
    container: DefaultContainer
    integration_manager: IntegrationManager

    def create_app(self, *, control_token: str):
        """Create the HTTP application around this prebuilt Host.

        The application reuses this exact container and delegates provider
        startup and shutdown to this Host's Integration Manager.

        Examples:
            Create an application for Uvicorn:
                ```python
                app = host.create_app(control_token=launch_token)
                ```

            Inspect the bound container:
                ```python
                app = host.create_app(control_token=launch_token)
                assert app.state.container is host.container
                ```

        Args:
            control_token: High-entropy per-launch supervisor token.

        Returns:
            FastAPI: Application bound to the verified immutable Host runtime.

        Notes:
            Calling this method does not start sockets or provider transports;
            FastAPI lifespan owns those operations.
        """

        from aethergraph.server.app_factory import create_app

        app = create_app(
            workspace=str(self.workspace),
            cfg=self.container.settings,
            container=self.container,
            integration_manager=self.integration_manager,
            deployment_mode=True,
        )
        from .control import install_host_control_routes

        install_host_control_routes(
            app=app,
            host=self,
            control_token=control_token,
        )
        return app


def build_host(
    *,
    manifest: HostManifest,
    runtime_identity: HostRuntimeIdentity,
    workspace: str | Path,
    settings: AppSettings,
    provider_connections: Sequence[HostProviderConnection] = (),
) -> AGHost:
    """Compose one immutable local AG Host from verified launch inputs.

    The builder validates manifest and compiler identities, creates one isolated
    registry and provider-neutral container, imports only the verified compiled
    entrypoint, validates graph/agent registration, installs immutable routes and
    operational stores, and constructs explicit provider lifecycle ownership.

    Examples:
        Build an endpoint-only Host:
            ```python
            host = build_host(
                manifest=manifest,
                runtime_identity=runtime_identity,
                workspace="deployment/runtime",
                settings=settings,
            )
            ```

        Build a Host with explicit providers:
            ```python
            host = build_host(
                manifest=manifest,
                runtime_identity=runtime_identity,
                workspace="deployment/runtime",
                settings=settings,
                provider_connections=(slack_provider,),
            )
            ```

    Args:
        manifest: Canonically sealed immutable launch manifest.
        runtime_identity: Supervisor-verified interpreter and settings identities.
        workspace: Exact deployment-owned operational workspace.
        settings: Pinned application settings selected by the control plane.
        provider_connections: Exact provider adapters and transport factories.

    Returns:
        AGHost: Fully composed Host ready for HTTP application startup.

    Notes:
        The builder performs no source discovery, release acquisition, dependency
        installation, environment credential lookup, or alternate graph loading.
    """

    expected_digest = compute_host_manifest_digest(manifest)
    if manifest.manifest_digest != expected_digest:
        raise HostManifestError("Host manifest digest does not match its canonical content.")
    _validate_runtime_identity(manifest, runtime_identity)
    inspection = validate_compiled_build(manifest)
    workspace_path = _prepare_workspace(Path(workspace), manifest)

    registry = UnifiedRegistry(allow_overwrite=False)
    set_current_registry(registry)
    pinned_settings = settings.model_copy(deep=True)
    pinned_settings.workspace = str(workspace_path)
    container = build_default_container(
        root=str(workspace_path),
        cfg=pinned_settings,
        channel_adapters={
            _provider_prefix(connection.integration_kind): connection.delivery_adapter
            for connection in provider_connections
        },
    )
    install_services(container)
    entrypoint = _load_entrypoint(
        container=container,
        build_root=Path(manifest.build_root or ""),
        package_name=inspection.manifest.package_name,
        module_name=manifest.entrypoint_module,
        symbol_name=manifest.entrypoint_symbol,
    )
    _validate_registration(container, manifest)

    connections = tuple(
        IntegrationConnection(
            integration_id=connection.integration_id,
            integration_kind=connection.integration_kind,
            transport=connection.transport_factory(container),
            delivery_adapter=connection.delivery_adapter,
            close_delivery=connection.close_delivery,
        )
        for connection in provider_connections
    )
    integration_manager = IntegrationManager(
        manifest=manifest,
        connections=connections,
    )
    install_integration_ingress(container=container, manifest=manifest)
    return AGHost(
        manifest=manifest,
        runtime_identity=runtime_identity,
        workspace=workspace_path,
        inspection=inspection,
        entrypoint=entrypoint,
        container=container,
        integration_manager=integration_manager,
    )


def _validate_runtime_identity(
    manifest: HostManifest,
    runtime_identity: HostRuntimeIdentity,
) -> None:
    expected = {
        "environment_snapshot_digest": manifest.environment_snapshot_digest,
        "runtime_profile_digest": manifest.runtime_profile_digest,
        "application_settings_digest": manifest.application_settings_digest,
    }
    actual = {
        "environment_snapshot_digest": runtime_identity.environment_snapshot_digest,
        "runtime_profile_digest": runtime_identity.runtime_profile_digest,
        "application_settings_digest": runtime_identity.application_settings_digest,
    }
    mismatched = sorted(name for name in expected if expected[name] != actual[name])
    if mismatched:
        raise HostManifestError("Runtime identity mismatch: " + ", ".join(mismatched))


def _prepare_workspace(path: Path, manifest: HostManifest) -> Path:
    workspace = path.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    marker_path = workspace / "host-workspace.json"
    marker = {
        "deployment_id": manifest.deployment_id,
        "workspace_identity": manifest.workspace_identity,
    }
    if marker_path.exists():
        try:
            existing = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HostManifestError("Deployment workspace marker is invalid.") from exc
        if existing != marker:
            raise HostManifestError("Deployment workspace belongs to another Host identity.")
    else:
        marker_path.write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    return workspace


def _load_entrypoint(
    *,
    container: DefaultContainer,
    build_root: Path,
    package_name: str,
    module_name: str,
    symbol_name: str,
):
    generated_src = build_root.resolve() / "src"
    if not generated_src.is_dir():
        raise HostManifestError("Compiled build is missing its verified src directory.")
    loaded = sorted(
        name for name in sys.modules if name == package_name or name.startswith(f"{package_name}.")
    )
    if loaded:
        raise HostManifestError("Compiled package was imported before Host verification.")
    source_path = str(generated_src)
    if source_path in sys.path:
        raise HostManifestError("Compiled source path was installed before Host verification.")
    sys.path.insert(0, source_path)
    try:
        with use_services(container):
            module = importlib.import_module(module_name)
    except Exception as exc:
        raise HostManifestError("Verified compiled entrypoint import failed.") from exc
    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(generated_src):
        raise HostManifestError("Compiled entrypoint resolved outside the verified build.")
    symbol = getattr(module, symbol_name, None)
    if not callable(symbol):
        raise HostManifestError("Compiled entrypoint symbol is not callable.")
    return symbol


def _validate_registration(container: DefaultContainer, manifest: HostManifest) -> None:
    try:
        container.registry.get_graphfn(name=manifest.graph_id)
        container.registry.get_agent(name=manifest.entry_agent_id)
    except KeyError as exc:
        raise HostManifestError(
            "Compiled entrypoint did not register the declared surface."
        ) from exc
    agent_meta = container.registry.get_meta(
        nspace="agent",
        name=manifest.entry_agent_id,
    )
    backing = (agent_meta or {}).get("backing", {})
    if backing.get("type") != "graphfn" or backing.get("name") != manifest.graph_id:
        raise HostManifestError("Compiled agent registration does not back the declared graph.")


def _provider_prefix(kind: IntegrationKind) -> str:
    prefixes = {
        IntegrationKind.SLACK: "slack",
        IntegrationKind.TELEGRAM: "tg",
    }
    try:
        return prefixes[kind]
    except KeyError as exc:
        raise HostManifestError(
            f"Provider connection kind is not transport-managed: {kind.value}"
        ) from exc


__all__ = [
    "AGHost",
    "HostProviderConnection",
    "HostRuntimeIdentity",
    "build_host",
]
