from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    flow_id: str | None = None  # optional flow ID within a graph -- not implemented yet

    # Tooling / proveance (optional)
    tool_name: str | None = None
    tool_version: str | None = None

    # Extra tags
    labels: dict[str, Any] = field(default_factory=dict)

    # Internal override for memory scope ID
    _memory_scope_id: str | None = None

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
        """Dimensions for MeteringService: what to attach to events."""
        out: dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.org_id:
            out["org_id"] = self.org_id
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
        if self.flow_id:
            out["flow_id"] = self.flow_id
        return out

    def with_memory_scope(self, mem_scope_id: str) -> Scope:
        """Return a copy with explicit memory scope override"""
        return replace(self, _memory_scope_id=mem_scope_id)

    def memory_scope_id(self) -> str:
        """
        Stable key for “memory bucket”.
        Default precedence: explicit override > session > user > run > org > app.
        """
        if self._memory_scope_id:
            return self._memory_scope_id
        if self.session_id:
            return f"session:{self.session_id}"
        if self.user_id:
            return f"user:{self.user_id}"
        if self.run_id:
            return f"run:{self.run_id}"
        if self.org_id:
            return f"org:{self.org_id}"
        if self.app_id:
            return f"app:{self.app_id}"
        return "global"

    def rag_labels(self, *, scope_id: str | None = None) -> dict[str, Any]:
        """
        Labels that should be stamped on RAG docs/chunks.
        scope_id is usually memory_scope_id (for memory-tied corpora),
        but can be any logical scope key.
        """
        out: dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.org_id:
            out["org_id"] = self.org_id
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
        if scope_id:
            out["scope_id"] = scope_id
        return out

    def rag_filter(self, *, scope_id: str | None = None) -> dict[str, Any]:
        """
        Default filter for RAG search based on identity.
        You can adjust strictness (e.g., ignore run_id for per-user corpora).
        """
        out: dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.org_id:
            out["org_id"] = self.org_id
        if scope_id:
            out["scope_id"] = scope_id
        # you can choose to include session_id / run_id only for very strict isolation
        return out
