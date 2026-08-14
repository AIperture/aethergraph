"""Stable contracts for embedding an AetherGraph runtime in another host."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aethergraph.config.config import AppSettings


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
class RuntimeSemanticEvent:
    """Host-facing semantic event with its durable shared-log cursor."""

    cursor: int
    kind: str
    event: Mapping[str, Any]


__all__ = [
    "RuntimeGraphRegistration",
    "RuntimeIdentity",
    "RuntimeModelProfile",
    "RuntimeOpenRequest",
    "RuntimeRunRecord",
    "RuntimeRunRequest",
    "RuntimeRunStatus",
    "RuntimeSemanticEvent",
]
