from __future__ import annotations

from dataclasses import dataclass

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.knowledge import KnowledgeBackend
from aethergraph.services.scope.scope_factory import ScopeFactory

from .node_kb import NodeKB


@dataclass
class KBFactory:
    """
    Factory to build NodeKB instances bound to a dedicated KB scope.

    Typically:
      - KB scope = scope_factory.for_kb(identity=request_identity)
    """

    backend: KnowledgeBackend
    scope_factory: ScopeFactory

    def for_identity(self, identity: RequestIdentity | None) -> NodeKB:
        kb_scope = self.scope_factory.for_kb(identity=identity)
        return NodeKB(backend=self.backend, scope=kb_scope)

    def for_scope(self, scope) -> NodeKB:
        """
        Optional helper if you ever want to bind KB to an explicit Scope
        (e.g. for tests).
        """
        return NodeKB(backend=self.backend, scope=scope)
