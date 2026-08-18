"""Authoritative AG Host composition services."""

from .builder import AGHost, HostProviderConnection, HostRuntimeIdentity, build_host
from .compatibility import build_release_compatibility
from .control import (
    HostDiagnostics,
    HostHealth,
    HostReadiness,
    install_host_control_routes,
)
from .endpoint_credentials import EndpointCredentialRegistry
from .manifest import (
    HostCompatibilityError,
    HostManifestError,
    compute_host_manifest_digest,
    load_host_manifest,
    seal_host_manifest,
    validate_compiled_build,
)

__all__ = [
    "AGHost",
    "HostProviderConnection",
    "HostCompatibilityError",
    "HostManifestError",
    "HostDiagnostics",
    "HostHealth",
    "HostReadiness",
    "HostRuntimeIdentity",
    "EndpointCredentialRegistry",
    "build_host",
    "build_release_compatibility",
    "compute_host_manifest_digest",
    "install_host_control_routes",
    "load_host_manifest",
    "seal_host_manifest",
    "validate_compiled_build",
]
