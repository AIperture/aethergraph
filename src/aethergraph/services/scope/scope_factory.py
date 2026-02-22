from dataclasses import dataclass, replace

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.services.scope.scope import Scope, ScopeLevel


@dataclass(frozen=True)
class ScopeFactory:
    def base_from_identity(self, identity: RequestIdentity | None) -> Scope:
        """
        Create a base Scope from a RequestIdentity.
        """
        if identity is None:
            return Scope(mode="local")

        return Scope(
            org_id=identity.org_id,
            user_id=identity.user_id,
            client_id=identity.client_id,
            mode=identity.mode,
        )

    def for_node(
        self,
        *,
        identity: RequestIdentity | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        session_id: str | None = None,
        app_id: str | None = None,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> Scope:
        """
        Create a Scope for a specific node execution.
        """
        base = self.base_from_identity(identity)
        return Scope(
            org_id=base.org_id,
            user_id=base.user_id,
            client_id=base.client_id,
            mode=base.mode,
            app_id=app_id or base.app_id,
            session_id=session_id,
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            tool_name=tool_name,
            tool_version=tool_version,
        )

    def for_run(
        self,
        *,
        identity: RequestIdentity | None = None,
        run_id: str,
        graph_id: str | None = None,
        session_id: str | None = None,
        flow_id: str | None = None,
    ) -> Scope:
        base = self.base_from_identity(identity)
        return Scope(
            org_id=base.org_id,
            user_id=base.user_id,
            client_id=base.client_id,
            mode=base.mode,
            app_id=base.app_id,
            session_id=session_id,
            run_id=run_id,
            graph_id=graph_id,
            node_id=None,
            flow_id=flow_id,
        )

    def for_memory(
        self,
        *,
        identity: RequestIdentity | None = None,
        run_id: str,
        graph_id: str | None = None,
        node_id: str | None = None,
        session_id: str | None = None,
        level: ScopeLevel | None = "session",
        custom_scope_id: str | None = None,
    ) -> Scope:
        """
        Build a Scope for MemoryFacade.

        Responsibilities:
          - Attach tenant/user/run/session/node identity.
          - Record the desired memory_level (run/session/user/org/scope).
          - Optionally apply a *full* custom memory scope override.

        Bucket string semantics (when no custom_scope_id is provided) now live
        in Scope.memory_scope_id(), which uses:
          - memory_level (if set), then
          - fallback precedence: session > user > run > org > app > global.
        """
        base = self.for_node(
            identity=identity,
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            session_id=session_id,
        )

        # Attach memory_level to the scope
        s = replace(base, memory_level=level)

        # If a custom_scope_id is provided, treat it as the canonical bucket ID.
        # This will become _memory_scope_id and override memory_scope_id().
        if custom_scope_id:
            return s.with_memory_scope(custom_scope_id, memory_level=level)

        return s

    def for_trigger(
        self,
        *,
        identity: RequestIdentity | None = None,
    ) -> Scope:
        """
        Build a Scope for TriggerFacade.

        For now, this is basically the same as base_from_identity, but we keep it separate in case trigger-specific logic emerges.
        """
        return self.base_from_identity(identity)

    def for_kb(
        self,
        *,
        identity: RequestIdentity | None = None,
        app_id: str | None = None,
    ) -> Scope:
        """
        Build a stable, user-level Scope for the knowledge base.

        - Ignores run/session/node so KB corpora naturally live across runs.
        - Sets memory_level='user' so memory_scope_id() is user-centric.
        """
        base = self.base_from_identity(identity)

        s = Scope(
            org_id=base.org_id,
            user_id=base.user_id,
            client_id=base.client_id,
            mode=base.mode,
            app_id=app_id or base.app_id or self.default_app_id,
            # NOTE: no session/run/graph/node here – KB is not run-scoped
        )

        # For KB we want a user-centric bucket by default
        # (org:user:user_id if org present; falls back to user/client).
        return replace(s, memory_level="user")
