from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.artifacts import AsyncArtifactStore  # generic protocol
from aethergraph.contracts.services.memory import HotLog, Indices, Persistence
from aethergraph.services.memory.facade import MemoryFacade

"""
    # --- Artifacts (async FS store)
    artifacts = FSArtifactStore(artifacts_dir)

    # --- KV for hotlog/indices (choose EphemeralKV or SQLiteKV)
    kv = SQLiteKV(f"{artifacts_dir}/kv.sqlite") if durable else EphemeralKV()

    # --- HotLog + Indices
    hotlog   = KVHotLog(kv, default_ttl_s=7*24*3600, default_limit=1000)
    indices  = KVIndices(kv, ttl_s=7*24*3600)

    # --- Persistence (JSONL under artifacts_dir/mem/<session>/events/...)
    persistence = FSPersistence(base_dir=artifacts_dir)

    # --- Factory
    factory = MemoryFactory(
        hotlog=hotlog,
        persistence=persistence,
        indices=indices,
        artifacts=artifacts,
        hot_limit=1000,
        hot_ttl_s=7*24*3600,
        default_signal_threshold=0.25,
    )

    # --- Global session handle (optional convenience)
    global_mem = factory.for_session("global", run_id="global")
"""


@dataclass(frozen=True)
class MemoryFactory:
    """Factory for creating MemoryFacade instances with shared components."""

    hotlog: HotLog
    persistence: Persistence
    indices: Indices  # key-value backed indices for fast lookups, not artifact storage index
    artifacts: AsyncArtifactStore
    hot_limit: int = 1000
    hot_ttl_s: int = 7 * 24 * 3600
    default_signal_threshold: float = 0.0
    logger: Any | None = None
    llm_service: Any | None = None  # LLMService
    rag_facade: Any | None = None  # RAGFacade

    def for_session(
        self,
        run_id: str,
        *,
        graph_id: str | None = None,
        node_id: str | None = None,
    ) -> MemoryFacade:
        return MemoryFacade(
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            hotlog=self.hotlog,
            persistence=self.persistence,
            indices=self.indices,
            artifact_store=self.artifacts,
            hot_limit=self.hot_limit,
            hot_ttl_s=self.hot_ttl_s,
            default_signal_threshold=self.default_signal_threshold,
            logger=self.logger,
            rag=self.rag_facade,
            llm=self.llm_service,
        )
