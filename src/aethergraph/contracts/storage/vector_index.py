from dataclasses import dataclass, field
from typing import Any, Protocol

"""
Vector index interface for storing and retrieving vector embeddings.

It can be used in rag services or any system that requires vector similarity search.
"""


class VectorIndex(Protocol):
    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metas: list[dict[str, Any]],
    ) -> None:
        """
        Insert or upsert vectors into a corpus.

        - corpus_id: logical collection name
        - chunk_ids: user IDs for each vector
        - vectors: len == len(chunk_ids), each a dense float vector
        - metas: arbitrary metadata (e.g. {"doc_id": ..., "offset": ...})
        """

    async def delete(
        self,
        corpus_id: str,
        chunk_ids: list[str] | None = None,
    ) -> None:
        """
        Delete entire corpus (chunk_ids=None) or specific chunks.
        """

    async def search(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
        where: dict[str, Any] | None = None,
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]: ...

    """
    where: equality filters on *promoted* fields (org_id, user_id, scope_id, etc.)
    max_candidates: limit number of candidate rows to score (after SQL WHERE, before cosine).
    """

    # Each dict MUST look like:
    # {"chunk_id": str, "score": float, "meta": dict[str, Any]}

    # Optional
    async def list_corpora(self) -> list[str]: ...
    async def list_chunks(self, corpus_id: str) -> list[str]: ...


PROMOTED_FIELDS = {
    "scope_id",
    "user_id",
    "org_id",
    "client_id",
    "session_id",
    "run_id",
    "graph_id",
    "node_id",
    "kind",
    "source",
}


@dataclass
class IndexMeta:
    """
    NOTE: When used with ScopedIndices, identity fields (scope_id, user_id, org_id, etc.)
    are usually left as None here and provided instead by scope.rag_labels(). In that
    common path, this struct is primarily for type-specific metadata (kind, source, ts, ...).
    """

    # tenant / scope
    scope_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    client_id: str | None = None
    session_id: str | None = None

    # run / graph context
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None

    # content type
    kind: str | None = None  # e.g. "artifact", "memory_event"
    source: str | None = None  # e.g. "hotlog", "artifact_index"

    # time
    ts: str | None = None  # human-readable ISO
    created_at_ts: float | None = None  # numeric, for DB index

    # free-form / extra labels
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scope_id": self.scope_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "client_id": self.client_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "source": self.source,
            "ts": self.ts,
            "created_at_ts": self.created_at_ts,
        }
        d.update(self.extra)
        # Strip Nones so meta stays compact
        return {k: v for k, v in d.items() if v is not None}
