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
from aethergraph.storage.vector_index.faiss_index import FAISSVectorIndex
from aethergraph.storage.vector_index.sqlite_index import SQLiteVectorIndex


def _build_vector_index(root: str, cfg: SearchBackendSettings) -> VectorIndex:
    if cfg.backend == "sqlite_vector":
        s = cfg.sqlite_vector
        index_root = os.path.join(root, s.dir)
        return SQLiteVectorIndex(root=index_root)

    if cfg.backend == "faiss_vector":
        s = cfg.faiss_vector
        index_root = os.path.join(root, s.dir)
        return FAISSVectorIndex(root=index_root, dim=s.dim)

    raise ValueError(f"unsupported active search backend {cfg.backend!r}")


def _build_lexical_index(root: str, cfg: SearchBackendSettings) -> SQLiteLexicalIndex:
    lcfg = cfg.sqlite_lexical
    db_path = os.path.join(root, lcfg.dir, lcfg.filename)
    return SQLiteLexicalIndex(root=db_path)


def _build_search_backend(
    *,
    root: str,
    settings: SearchBackendSettings,
    embedder: EmbeddingClientProtocol | None,
) -> SearchBackend:
    scfg = settings
    if scfg.backend in ("sqlite_vector", "faiss_vector"):
        if embedder is None:
            raise RuntimeError(
                f"Search backend {scfg.backend!r} requires an embedding client. "
                "Configure the required embedding dependency before runtime composition."
            )

        index = _build_vector_index(root, scfg)

        lexical_index = None
        if scfg.enable_lexical:
            lexical_index = _build_lexical_index(root, scfg)

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
    """Build the required legacy-active backend used by `ScopedIndices`.

    Only the two implemented vector-backed choices remain before the S9 canonical
    composition cut. Missing embeddings fail construction; no null backend or
    backend retry is available.

    Examples:
        Build the configured SQLite-backed search service:
            ```python
            backend = build_search_backend(settings, embedder=embedding_client)
            ```

        Observe missing required capability:
            ```python
            with pytest.raises(RuntimeError):
                build_search_backend(settings, embedder=None)
            ```

    Args:
        cfg: Active application settings containing the exact legacy search choice.
        embedder: Required embedding dependency for either implemented backend.

    Returns:
        SearchBackend: Configured pre-S9 `ScopedIndices` backend.

    Notes:
        This function remains only until the atomic canonical provider binding. It
        never returns an empty/null implementation and never changes backend choice.
    """
    root = os.path.abspath(cfg.workspace)
    return _build_search_backend(
        root=root,
        settings=cfg.search,
        embedder=embedder,
    )
