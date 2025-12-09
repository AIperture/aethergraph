from dataclasses import dataclass
from typing import Literal

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.services.scope.scope import Scope


@dataclass(frozen=True)
class ScopeFactory:
    default_app_id: str | None = None

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
            app_id=self.default_app_id,
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
        s = self.base_from_identity(identity)
        s.run_id = run_id
        s.graph_id = graph_id
        s.session_id = session_id
        s.flow_id = flow_id
        return s

    def for_memory(
        self,
        *,
        identity: RequestIdentity | None = None,
        run_id: str,
        graph_id: str | None = None,
        node_id: str | None = None,
        session_id: str | None = None,
        level: Literal["session", "user", "run", "org"] = "session",
        custom_scope_id: str | None = None,
    ):
        """
        Scope for MemoryFacade. level defines how we group memory:
        - "session": per-session (default)
        - "user":    across runs/sessions for a given user
        - "run":     per-run
        - "org":     org-level memory
        """
        s = self.for_node(
            identity=identity,
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            session_id=session_id,
        )
        if custom_scope_id is not None:
            mem_id = custom_scope_id
        else:
            if level == "session":
                base = session_id or run_id
                mem_id = f"session:{base}"
            elif level == "user":
                u = s.user_id or s.client_id or "anon"
                mem_id = f"user:{u}"
            elif level == "run":
                mem_id = f"run:{run_id}"
            elif level == "org":
                o = s.org_id or "orgless"
                mem_id = f"org:{o}"
            else:  # pragma: no cover
                mem_id = f"run:{run_id}"

        return s.with_memory_scope(mem_id)
