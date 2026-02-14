# search_factory.py

from __future__ import annotations

import os

from aethergraph.config.config import AppSettings
from aethergraph.config.search import SearchBackendSettings
from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.storage.search_backend import SearchBackend
from aethergraph.contracts.storage.vector_index import VectorIndex
from aethergraph.storage.lexical_index.sqlite_lexical_index import SQLiteLexicalIndex
from aethergraph.storage.search_backend.generic_backend import GenericSearchBackend
from aethergraph.storage.search_backend.null_backend import NullSearchBackend
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


def build_lexical_index_for_search(root: str, cfg: SearchBackendSettings) -> SQLiteLexicalIndex:
    """
    Helper to build a SQLite-based lexical index for search.

    This is used in two contexts:
      - When cfg.backend == "sqlite_lexical": a pure lexical SearchBackend.
      - When cfg.backend is a vector backend *and* cfg.enable_lexical is True:
        attach the lexical index to GenericSearchBackend for hybrid modes.
    """
    lcfg = cfg.sqlite_lexical
    db_path = os.path.join(root, lcfg.dir, lcfg.filename)
    return SQLiteLexicalIndex(root=db_path)


# def build_search_backend_from_settings(
#     *,
#     root: str,
#     settings: SearchBackendSettings,
#     embedder: EmbeddingClientProtocol | None,
# ) -> SearchBackend:
#     """
#     Build a SearchBackend from a SearchBackendSettings block.

#     This is the generic builder used by indices, KB, etc.
#     """
#     scfg = settings

#     # 1) No search at all
#     if scfg.backend == "none":
#         return NullSearchBackend()

#     # 2) Pure lexical, no LLM / embeddings
#     if scfg.backend == "sqlite_lexical":
#         lcfg = scfg.sqlite_lexical
#         db_path = os.path.join(root, lcfg.dir, lcfg.filename)
#         return SQLiteLexicalSearchBackend(db_path=db_path)

#     # 3) Vector search backends (sqlite or faiss)
#     if scfg.backend in ("sqlite_vector", "faiss_vector"):
#         if embedder is None:
#             raise RuntimeError(
#                 f"Search backend {scfg.backend!r} requires an embedding client. "
#                 "Pass an EmbeddingClientProtocol instance into build_search_backend_from_settings()."
#             )

#         index = build_vector_index_for_search(root, scfg)
#         return GenericVectorSearchBackend(index=index, embedder=embedder)

#     raise ValueError(f"Unknown search backend: {scfg.backend!r}")


def build_search_backend_from_settings(
    *,
    root: str,
    settings: SearchBackendSettings,
    embedder: EmbeddingClientProtocol | None,
) -> SearchBackend:
    """
    Build a SearchBackend from a SearchBackendSettings block.

    This is the generic builder used by indices, KB, etc.
    """
    scfg = settings

    # 1) No search at all
    if scfg.backend == "none":
        return NullSearchBackend()

    # 2) Pure lexical, no LLM / embeddings
    if scfg.backend == "sqlite_lexical":
        raise NotImplementedError(
            "The 'sqlite_lexical' backend is currently only supported as an optional lexical index attached to vector backends. Support for a pure lexical SearchBackend will be added in a future release."
        )

    # 3) Vector search backends (sqlite or faiss), with optional lexical index
    if scfg.backend in ("sqlite_vector", "faiss_vector"):
        if embedder is None:
            raise RuntimeError(
                f"Search backend {scfg.backend!r} requires an embedding client. "
                "Pass an EmbeddingClientProtocol instance into build_search_backend_from_settings()."
            )

        index = build_vector_index_for_search(root, scfg)

        lexical_index = None
        if scfg.enable_lexical:
            lexical_index = build_lexical_index_for_search(root, scfg)

        # which now supports an optional lexical index and a `mode` parameter.
        return GenericSearchBackend(
            index=index,
            embedder=embedder,
            lexical=lexical_index,
        )

    raise ValueError(f"Unknown search backend: {scfg.backend!r}")


def build_search_backend(
    cfg: AppSettings,
    *,
    embedder: EmbeddingClientProtocol | None,
) -> SearchBackend:
    """
    Factory to build the high-level SearchBackend used by ScopedIndices.

    Respects cfg.search.backend:
      - "none"           -> NullSearchBackend
      - "sqlite_lexical" -> SQLiteLexicalSearchBackend (pure lexical FTS)
      - "sqlite_vector"  -> GenericSearchBackend + SQLiteVectorIndex
                           (+ optional SQLite lexical index if cfg.search.enable_lexical)
      - "faiss_vector"   -> GenericSearchBackend + FAISSVectorIndex
                           (+ optional SQLite lexical index if cfg.search.enable_lexical)
    """
    root = os.path.abspath(cfg.root)
    return build_search_backend_from_settings(
        root=root,
        settings=cfg.search,
        embedder=embedder,
    )


def build_kb_search_backend(
    cfg: AppSettings,
    *,
    embedder: EmbeddingClientProtocol | None,
) -> SearchBackend:
    """
    Build the SearchBackend used by the KnowledgeBackend / KB.
    """
    root = os.path.abspath(cfg.root)
    return build_search_backend_from_settings(
        root=root,
        settings=cfg.knowledge.search,
        embedder=embedder,
    )
