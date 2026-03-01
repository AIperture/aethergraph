from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

ScopeLevel = Literal["scope", "session", "run", "user", "org"]


@dataclass(frozen=True)
class Scope:
    # Tenant / actor
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    mode: str | None = None  # "cloud", "demo", "local", etc.

    # App / execution context
    app_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    flow_id: str | None = None  # optional flow ID within a graph -- not implemented yet

    # Tooling / proveance (to delete or move later)
    tool_name: str | None = None
    tool_version: str | None = None

    # Extra tags (to delete or move later)
    labels: dict[str, Any] = field(default_factory=dict)

    # logical memory level (scope/session/run/user/org); if None, will be inferred from other fields
    memory_level: ScopeLevel | None = None
    # Internal override for memory scope ID
    _memory_scope_id: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def identity_labels(self) -> dict[str, str]:
        """
        Canonical identity labels shared across memory, artifacts, and metering.
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
        if self.agent_id:
            out["agent_id"] = self.agent_id
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

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def artifact_scope_labels(self) -> dict[str, str]:
        """
        Labels to attach to every artifact for this scope.
        These will be mirrored both into Artifact.labels and the index.
        """
        out: dict[str, str] = {}
        out.update(self.identity_labels())
        scope_id = self.memory_scope_id()
        if scope_id:
            out["scope_id"] = scope_id
        return out

    def metering_dimensions(self) -> dict[str, Any]:
        """Dimensions for MeteringService."""
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

    def with_memory_scope(self, mem_scope_id: str, memory_level: ScopeLevel | None = None) -> Scope:
        """Return a copy with explicit memory scope override"""
        return replace(
            self,
            _memory_scope_id=mem_scope_id,
            memory_level=memory_level if memory_level is not None else self.memory_level,
        )

    def memory_scope_id(self) -> str:
        """
        Stable key for “memory bucket”.

        If memory_level is set, we derive scope ID from it:
        - "session" -> session:<session_id>
        - "user"   -> user:<user_id>
        - "run"    -> run:<run_id>
        - "org"    -> org:<org_id>
        - "scope"  -> global

        Otherwise, fall back to precedence:
          override > session > user > run > org > app > global
        """
        # 1) Explicit override always wins
        if self._memory_scope_id:
            return self._memory_scope_id

        # 2) If we know the logical memory level, honor it
        lvl = self.memory_level
        if lvl == "session" and self.session_id:
            # could be "session:<session_id>" or richer like "org:...:user:...:session:..."
            return f"session:{self.session_id}"
        if lvl == "user":
            if self.org_id and self.user_id:
                return f"org:{self.org_id}:user:{self.user_id}"
            if self.user_id:
                return f"user:{self.user_id}"
            if self.client_id:
                return f"user:{self.client_id}"
            return "user:anon"
        if lvl == "run" and self.run_id:
            return f"run:{self.run_id}"
        if lvl == "org" and self.org_id:
            return f"org:{self.org_id}"
        if lvl == "scope":
            return "global"

        # 3) Fallback to old precedence for back-compat
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
        if scope_id is None:
            scope_id = self.memory_scope_id()

        out: dict[str, Any] = {}
        out.update(self.identity_labels())
        if scope_id:
            out["scope_id"] = scope_id
        return out

    def rag_filter(self, *, scope_id: str | None = None) -> dict[str, Any]:
        """
        Default filter for RAG search based on identity.

        By default we isolate on:
        - org_id (tenant)
        - user_id (actor within tenant)
        - scope_id (memory bucket, usually memory_scope_id)
        """
        if scope_id is None:
            scope_id = self.memory_scope_id()

        out: dict[str, Any] = {}
        if self.user_id:
            out["user_id"] = self.user_id
        if self.org_id:
            out["org_id"] = self.org_id
        if scope_id:
            out["scope_id"] = scope_id
        return out

    def kb_scope_id(self) -> str:
        """
        Stable key for knowledge base buckets.

        Unlike memory_scope_id(), this is intentionally *not* session/run-scoped.
        It's primarily org+user (or client) based so KB persists across sessions.
        """
        if self.org_id and (self.user_id or self.client_id):
            u = self.user_id or self.client_id
            return f"org:{self.org_id}:user:{u}:kb"
        if self.user_id:
            return f"user:{self.user_id}:kb"
        if self.client_id:
            return f"user:{self.client_id}:kb"
        return "kb:global"

    def kb_index_labels(self) -> dict[str, Any]:
        """
        Labels we want to push into the KB vector index.

        We deliberately omit session_id/run_id so KB is user-level by default.
        """
        out: dict[str, Any] = {}
        if self.org_id:
            out["org_id"] = self.org_id
        u = self.user_id or self.client_id
        if u:
            out["user_id"] = u
        out["kb_scope_id"] = self.kb_scope_id()
        return out

    def kb_filter(self) -> dict[str, Any]:
        """
        Default filter for KB searches.

        Typically org_id + user_id + kb_scope_id so each user sees their KB.
        """
        out = self.kb_index_labels()
        return {k: v for k, v in out.items() if v is not None}
