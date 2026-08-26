"""Authoritative composition of one immutable local AG Host runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from aethergraph.config.config import AppSettings
from aethergraph.contracts.integration import HostManifest, IntegrationKind, ReleaseDependency
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

from .compatibility import (
    ENTRYPOINT_INPUT_SCHEMA,
    ENTRYPOINT_OUTPUT_SCHEMA,
    HOST_CAPABILITIES,
    HOST_SERVICES,
)
from .manifest import (
    HostCompatibilityError,
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
    aethergraph_version: str
    engine_version: str
    python_abi: str
    platform: str
    architecture: str
    dependency_lock_digest: str


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

        from .endpoint_credentials import EndpointCredentialRegistry

        endpoint_credentials = EndpointCredentialRegistry.from_manifest(self.manifest)

        app = create_app(
            workspace=str(self.workspace),
            cfg=self.container.settings,
            container=self.container,
            integration_manager=self.integration_manager,
            deployment_mode=True,
        )
        app.state.endpoint_credentials = endpoint_credentials
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
    _validate_release_compatibility(manifest, runtime_identity)
    inspection = validate_compiled_build(manifest)
    workspace_path = _prepare_workspace(Path(workspace))

    registry = UnifiedRegistry(allow_overwrite=False)
    set_current_registry(registry)
    pinned_settings = settings.model_copy(deep=True)
    pinned_settings.workspace = str(workspace_path)
    _bind_runtime_profile(pinned_settings, manifest.runtime_profile_name)
    container = build_default_container(
        root=str(workspace_path),
        cfg=pinned_settings,
        channel_adapters={
            _provider_prefix(connection.integration_kind): connection.delivery_adapter
            for connection in provider_connections
        },
        workspace_id=_host_storage_workspace_id(manifest),
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


def _bind_runtime_profile(settings: AppSettings, profile_name: str | None) -> None:
    """Make the manifest-selected Chat profile the Host application default.

    Intro:
        Studio and other control planes can select one named profile while the
        Host still receives the complete immutable settings snapshot.

    Examples:
        Bind a named profile:
            ```python
            _bind_runtime_profile(settings, "deployment")
            ```

        Preserve legacy manifests:
            ```python
            _bind_runtime_profile(settings, None)
            ```

    Args:
        settings: Deep-copied settings owned by this Host construction.
        profile_name: Optional manifest-pinned Chat profile name.

    Returns:
        None: The copied default profile is updated in place when selected.

    Notes:
        Missing named profiles fail closed. Credentials remain sourced from the
        immutable settings snapshot and are not copied into the manifest.
    """

    if profile_name is None:
        return
    profiles = {"default": settings.llm.default, **settings.llm.profiles}
    selected = profiles.get(profile_name)
    if selected is None:
        raise HostManifestError(
            f"Pinned runtime profile {profile_name!r} is missing from Host settings."
        )
    settings.llm.default = selected.model_copy(deep=True)


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


def _validate_release_compatibility(
    manifest: HostManifest,
    runtime_identity: HostRuntimeIdentity,
) -> None:
    compatibility = manifest.release_compatibility
    try:
        installed_engine = importlib.metadata.version("aethergraph-engine")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HostCompatibilityError(
            "Release requires aethergraph-engine but it is not installed; "
            "select the pinned interpreter or rebuild the release."
        ) from exc
    from aethergraph import __version__ as installed_ag

    installed_lock = tuple(
        _installed_dependency(name) for name in ("aethergraph", "aethergraph-engine")
    )
    installed_lock_digest = sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in installed_lock],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if compatibility.dependency_lock != installed_lock:
        raise HostCompatibilityError(
            "Release dependency lock does not match installed Host distributions; "
            "select the pinned interpreter or rebuild the release."
        )
    expected = {
        "aethergraph_version": compatibility.aethergraph_version,
        "engine_version": compatibility.engine_version,
        "python_abi": compatibility.python_abi,
        "platform": compatibility.platform,
        "architecture": compatibility.architecture,
        "dependency_lock_digest": compatibility.dependency_lock_digest,
    }
    actual = {
        "aethergraph_version": installed_ag,
        "engine_version": installed_engine,
        "python_abi": sys.implementation.cache_tag or "unknown",
        "platform": sys.platform,
        "architecture": platform.machine() or platform.architecture()[0],
        "dependency_lock_digest": installed_lock_digest,
    }
    runtime = {
        "aethergraph_version": runtime_identity.aethergraph_version,
        "engine_version": runtime_identity.engine_version,
        "python_abi": runtime_identity.python_abi,
        "platform": runtime_identity.platform,
        "architecture": runtime_identity.architecture,
        "dependency_lock_digest": runtime_identity.dependency_lock_digest,
    }
    mismatched = sorted(
        name
        for name in expected
        if expected[name] != actual[name] or expected[name] != runtime[name]
    )
    if mismatched:
        detail = ", ".join(
            f"{name} requires {expected[name]!r} but Host has {actual[name]!r}"
            for name in mismatched
        )
        raise HostCompatibilityError(
            "Release compatibility mismatch: "
            + detail
            + "; select the pinned interpreter or rebuild the release."
        )
    missing_capabilities = sorted(
        set(compatibility.host_capability_requirements) - HOST_CAPABILITIES
    )
    if missing_capabilities:
        raise HostCompatibilityError(
            "Host lacks required capabilities: "
            + ", ".join(missing_capabilities)
            + "; upgrade the Host or rebuild the release."
        )
    missing_services = sorted(set(compatibility.service_requirements) - HOST_SERVICES)
    if missing_services:
        raise HostCompatibilityError(
            "Host lacks required services: "
            + ", ".join(missing_services)
            + "; upgrade the Host or rebuild the release."
        )
    if compatibility.ingress_protocol_version != manifest.ingress_protocol_version:
        raise HostCompatibilityError("Release ingress protocol does not match Host manifest.")
    if compatibility.semantic_event_protocol_version != manifest.semantic_event_protocol_version:
        raise HostCompatibilityError("Release semantic protocol does not match Host manifest.")
    if compatibility.logical_output_requirements != ("origin",):
        raise HostCompatibilityError("Release requires unsupported logical outputs.")
    if compatibility.entrypoint_input_schema != ENTRYPOINT_INPUT_SCHEMA:
        raise HostCompatibilityError("Release entrypoint input schema is unsupported.")
    if compatibility.entrypoint_output_schema != ENTRYPOINT_OUTPUT_SCHEMA:
        raise HostCompatibilityError("Release entrypoint output schema is unsupported.")


def _installed_dependency(name: str) -> ReleaseDependency:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise HostCompatibilityError(
            f"Release dependency is not installed: {name}; "
            "select the pinned interpreter or rebuild the release."
        ) from exc
    record = distribution.read_text("RECORD")
    if not record:
        raise HostCompatibilityError(
            f"Release dependency has no immutable RECORD: {name}; rebuild the environment."
        )
    return ReleaseDependency(
        name=name,
        version=distribution.version,
        content_sha256=sha256(record.encode("utf-8")).hexdigest(),
    )


def _prepare_workspace(path: Path) -> Path:
    workspace = path.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _host_storage_workspace_id(manifest: HostManifest) -> str:
    identity = f"{manifest.deployment_id}\0{manifest.workspace_identity}".encode()
    return "host-workspace-" + sha256(identity).hexdigest()[:32]


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
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise ModuleNotFoundError(
            "Compiled entrypoint requires missing Python module "
            f"{missing!r}. Install the project dependencies in the selected runtime.",
            name=missing,
        ) from exc
    except ImportError as exc:
        detail = " ".join(str(exc).split())[:500] or "no import detail was provided"
        raise ImportError(f"Compiled entrypoint import failed: {detail}") from exc
    except Exception as exc:
        detail = " ".join(str(exc).split())[:500] or "no exception detail was provided"
        raise HostManifestError(
            "Verified compiled entrypoint import failed " f"({type(exc).__name__}): {detail}"
        ) from exc
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
