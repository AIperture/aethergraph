"""Stable contracts for embedding an AetherGraph runtime in another host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aethergraph.config.config import AppSettings
from aethergraph.storage.contracts import (
    StorageProviderSelection,
    StorageScope,
    StorageSecretResolver,
)
from aethergraph.storage.provider_registry import StorageProviderFactory


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Identity attached to one embedded runtime operation."""

    user_id: str
    org_id: str
    mode: str = "local"


@dataclass(frozen=True, slots=True)
class RuntimeOpenRequest:
    """Closed configuration used to construct one embedded runtime."""

    root: Path
    settings: AppSettings
    channel_adapters: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    workspace_id: str | None = None
    owner_scope: StorageScope | None = None
    storage_selection: StorageProviderSelection | None = None
    storage_providers: Mapping[str, StorageProviderFactory] = field(default_factory=dict)
    storage_secrets: StorageSecretResolver | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_adapters", MappingProxyType(dict(self.channel_adapters)))
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))
        object.__setattr__(
            self, "storage_providers", MappingProxyType(dict(self.storage_providers))
        )
        if self.workspace_id is not None and (
            not self.workspace_id.strip() or self.workspace_id != self.workspace_id.strip()
        ):
            raise ValueError("workspace_id must be exact and non-empty when supplied")


@dataclass(frozen=True, slots=True)
class RuntimeRunRequest:
    """Provider-neutral request to submit one graph run."""

    graph_id: str
    inputs: Mapping[str, Any]
    run_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    identity: RuntimeIdentity = field(
        default_factory=lambda: RuntimeIdentity(user_id="local", org_id="local")
    )
    origin: str = "local"
    agent_id: str | None = None
    app_id: str | None = None
    app_name: str | None = None
    run_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeRunRecord:
    """Stable host view of persisted run metadata."""

    run_id: str
    graph_id: str
    session_id: str | None
    status: str
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeRunStatus:
    """One run status response with optional durable output and diagnostics."""

    record: RuntimeRunRecord
    output: Mapping[str, Any] | None
    run_error_info: Mapping[str, Any] | None
    node_diagnostics: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RuntimeGraphRegistration:
    """Validated graph registration loaded inside an embedded runtime."""

    module_name: str
    symbol_name: str
    graph_id: str


@dataclass(frozen=True, slots=True)
class RuntimeModelProfile:
    """Immutable model-profile values exposed to an embedding Host."""

    name: str
    provider: str | None
    model: str | None


@dataclass(frozen=True, slots=True)
class RuntimeRegistrationSnapshot:
    """Immutable registration facts requested by an embedding Host."""

    agent_metadata: Mapping[str, Mapping[str, Any] | None]
    registered_graph_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RuntimeArtifactScope:
    """Provider-neutral scope for staging one Host resource."""

    source: str
    session_id: str | None = None
    run_id: str | None = None
    channel_key: str | None = None
    conversation_id: str | None = None
    graph_id: str = "channel"
    node_id: str = "resource_ingress"
    tool_name: str = "channel.resource_ingress"
    tool_version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class RuntimeStagedArtifact:
    """Immutable result of staging Host-authenticated bytes."""

    artifact_id: str
    size_bytes: int
    uri: str | None


@dataclass(frozen=True, slots=True)
class RuntimeArtifactRecord:
    """Immutable artifact metadata exposed for Host authorization."""

    artifact_id: str
    uri: str | None
    name: str | None
    mime: str | None
    size_bytes: int
    sha256: str | None
    org_id: str | None
    labels: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeSemanticEvent:
    """Host-facing semantic event with its durable shared-log cursor."""

    cursor: int
    kind: str
    event: Mapping[str, Any]


__all__ = [
    "RuntimeGraphRegistration",
    "RuntimeArtifactRecord",
    "RuntimeArtifactScope",
    "RuntimeIdentity",
    "RuntimeModelProfile",
    "RuntimeOpenRequest",
    "RuntimeRunRecord",
    "RuntimeRunRequest",
    "RuntimeRunStatus",
    "RuntimeRegistrationSnapshot",
    "RuntimeSemanticEvent",
    "RuntimeStagedArtifact",
]
