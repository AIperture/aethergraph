from typing import Any

from aethergraph.contracts.storage.vector_index import IndexMeta


def build_index_meta_from_scope(
    *,
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

    return m.to_dict()
