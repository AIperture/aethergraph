from dataclasses import dataclass

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
