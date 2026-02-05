# search_factory.py

from __future__ import annotations

import os

from aethergraph.config.config import AppSettings
from aethergraph.config.search import SearchBackendSettings
from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.storage.search_backend import SearchBackend
from aethergraph.contracts.storage.vector_index import VectorIndex
from aethergraph.storage.search_backend.generic_vector_backend import GenericVectorSearchBackend
from aethergraph.storage.search_backend.null_backend import NullSearchBackend
from aethergraph.storage.search_backend.sqlite_lexical_backend import SQLiteLexicalSearchBackend
from aethergraph.storage.vector_index.faiss_index import FAISSVectorIndex
from aethergraph.storage.vector_index.sqlite_index import SQLiteVectorIndex


def build_vector_index_for_search(root: str, cfg: SearchBackendSettings) -> VectorIndex:
    """
    Helper to build a VectorIndex specifically for search, based on cfg.search.backend.
    This is intentionally separate from storage.vector_index (legacy RAG index).
    """
    if cfg.backend == "sqlite_vector":
        s = cfg.sqlite_vector
        index_root = os.path.join(root, s.dir)
        return SQLiteVectorIndex(root=index_root)

    if cfg.backend == "faiss_vector":
        s = cfg.faiss_vector
        index_root = os.path.join(root, s.dir)
        return FAISSVectorIndex(root=index_root, dim=s.dim)

    raise ValueError(f"build_vector_index_for_search: unsupported backend {cfg.backend!r}")


def build_search_backend(
    cfg: AppSettings,
    *,
    embedder: EmbeddingClientProtocol | None,
) -> SearchBackend:
    """
    Factory to build the high-level SearchBackend used by ScopedIndices.

    Respects cfg.search.backend:
      - "none"          -> NullSearchBackend
      - "sqlite_lexical"-> SQLiteLexicalSearchBackend
      - "sqlite_vector" -> VectorSearchBackend + SQLiteVectorIndex
      - "faiss_vector"  -> VectorSearchBackend + FAISSVectorIndex
    """
    scfg = cfg.search
    root = os.path.abspath(cfg.root)

    # 1) No search at all
    if scfg.backend == "none":
        return NullSearchBackend()

    # 2) Pure lexical, no LLM / embeddings
    if scfg.backend == "sqlite_lexical":
        lcfg = scfg.sqlite_lexical
        db_path = os.path.join(root, lcfg.dir, lcfg.filename)
        return SQLiteLexicalSearchBackend(db_path=db_path)

    # 3) Vector search backends (sqlite or faiss)
    if scfg.backend in ("sqlite_vector", "faiss_vector"):
        if embedder is None:
            raise RuntimeError(
                f"Search backend {scfg.backend!r} requires an embedding client. "
                "Pass an EmbeddingClientProtocol instance into build_search_backend()."
            )

        index = build_vector_index_for_search(root, scfg)
        return GenericVectorSearchBackend(index=index, embedder=embedder)

    raise ValueError(f"Unknown search backend: {scfg.backend!r}")
