from __future__ import annotations
from venv import logger

from aethergraph.services.memory.hotlog_kv import KVHotLog
from aethergraph.services.memory.persist_fs import FSPersistence
from aethergraph.services.memory.indices import KVIndices
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.kv.factory import make_kv
from aethergraph.services.artifacts.factory import make_artifact_store

from dataclasses import dataclass
from typing import Any, Optional
from .facade import MemoryFacade
from aethergraph.contracts.services.memory import HotLog, Persistence, Indices
from aethergraph.contracts.services.artifacts import AsyncArtifactStore  # generic protocol

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
    """ Factory for creating MemoryFacade instances with shared components. """
    hotlog: HotLog
    persistence: Persistence
    indices: Indices # key-value backed indices for fast lookups, not artifact storage index
    artifacts: AsyncArtifactStore
    hot_limit: int = 1000
    hot_ttl_s: int = 7 * 24 * 3600
    default_signal_threshold: float = 0.25
    logger: Optional[Any] = None
    llm_service: Optional[Any] = None  # LLMService
    rag_facade: Optional[Any] = None  # RAGFacade

    def for_session(
        self,
        run_id: str,
        *,
        graph_id: Optional[str] = None,
        node_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> MemoryFacade:
        return MemoryFacade(
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            agent_id=agent_id,
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
