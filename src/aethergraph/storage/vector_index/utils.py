from typing import Any

from aethergraph.contracts.storage.vector_index import IndexMeta
from aethergraph.services.scope.scope import Scope


def build_index_meta_from_scope(
    *,
    scope: Scope | None,
    kind: str | None,
    source: str | None,
    ts: str | None,
    created_at_ts: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    m = IndexMeta(
        kind=kind,
        source=source,
        ts=ts,
        created_at_ts=created_at_ts,
        extra=extra or {},
    )
    if scope is not None:
        dims = scope.metering_dimensions()
        m.scope_id = scope.memory_scope_id()
        m.user_id = dims.get("user_id")
        m.org_id = dims.get("org_id")
        m.client_id = dims.get("client_id")
        m.session_id = dims.get("session_id")
        m.run_id = dims.get("run_id")
        m.graph_id = dims.get("graph_id")
        m.node_id = dims.get("node_id")

    return m.to_dict()
