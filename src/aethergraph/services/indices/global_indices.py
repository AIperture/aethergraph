from dataclasses import dataclass

from aethergraph.contracts.storage.search_backend import SearchBackend
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.scope.scope import Scope


@dataclass
class GlobalIndices:
    backend: SearchBackend

    def for_scope(
        self,
        scope: Scope,
        scope_id: str | None = None,
    ) -> ScopedIndices:
        return ScopedIndices(
            backend=self.backend,
            scope=scope,
            scope_id=scope_id,
        )
