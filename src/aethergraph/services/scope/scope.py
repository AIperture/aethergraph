from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scope:
    # Tenant / actor
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    mode: str | None = None  # "cloud", "demo", "local", etc.

    # App / execution context
    app_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None

    # Tooling / proveance (optional)
    tool_name: str | None = None
    tool_version: str | None = None

    # Extra tags
    labels: dict[str, Any] = field(default_factory=dict)

    def __item__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def artifact_scope_labels(self) -> dict[str, str]:
        """
        Labels to attach to every artifact for this scope.
        These will be mirrored both into Artifact.labels and the index.
        """
        out: dict[str, str] = {}
        if self.org_id:
            out["org_id"] = self.org_id
        if self.user_id:
            out["user_id"] = self.user_id
        if self.client_id:
            out["client_id"] = self.client_id
        if self.app_id:
            out["app_id"] = self.app_id
        if self.session_id:
            out["session_id"] = self.session_id
        if self.run_id:
            out["run_id"] = self.run_id
        if self.graph_id:
            out["graph_id"] = self.graph_id
        if self.node_id:
            out["node_id"] = self.node_id
        return out

    def metering_dimensions(self) -> dict[str, Any]:
        """
        Common dimensions to attach to metering events for this scope.
        """
        out: dict[str, Any] = {
            "org_id": self.org_id,
            "user_id": self.user_id or self.client_id,
            "client_id": self.client_id,
            "app_id": self.app_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
        }
        return out
