# search_settings.py (or wherever you keep config models)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .storage import FAISSVectorIndexSettings, SQLiteVectorIndexSettings

# ^ or wherever those two are defined


class SQLiteLexicalSearchSettings(BaseModel):
    """
    Settings for SQLite-based lexical search backend.
    Paths are relative to AppSettings.workspace.
    """

    dir: str = "search/sqlite_lexical"
    filename: str = "index.sqlite"


class SearchBackendSettings(BaseModel):
    """
    Config for the high-level SearchBackend used by ScopedIndices.

    backend:
      - "none"          -> NullSearchBackend (no search at all)
      - "sqlite_vector" -> VectorSearchBackend + SQLiteVectorIndex
      - "faiss_vector"  -> VectorSearchBackend + FAISSVectorIndex
      - "sqlite_lexical"-> SQLiteLexicalSearchBackend (no embeddings)
    """

    backend: Literal["none", "sqlite_vector", "faiss_vector", "sqlite_lexical"] = "sqlite_vector"

    # Vector search backends (reuse your existing index settings types,
    # but point them to search-specific directories by default).
    sqlite_vector: SQLiteVectorIndexSettings = SQLiteVectorIndexSettings(
        dir="search/vector_sqlite",
        filename="index.sqlite",
    )
    faiss_vector: FAISSVectorIndexSettings = FAISSVectorIndexSettings(
        dir="search/vector_faiss",
        dim=None,
    )

    # Lexical search backend (for pure sqlite_lexical backend, *and* for
    # optional lexical index when enable_lexical=True on vector backends).
    sqlite_lexical: SQLiteLexicalSearchSettings = SQLiteLexicalSearchSettings()

    # NEW: toggle lexical index when using vector backends
    enable_lexical: bool = True


class KnowledgeSettings(BaseModel):
    """
    Settings for the Knowledge Base subsystem.
    """

    # Where LocalFSKnowledgeBackend stores corpus files
    corpus_root: str = "kb/corpora"

    # Search backend for KB (separate index from global indices)
    search: SearchBackendSettings = SearchBackendSettings(
        backend="sqlite_vector",
        sqlite_vector=SQLiteVectorIndexSettings(
            dir="kb/search/vector_sqlite",
            filename="index.sqlite",
        ),
        sqlite_lexical=SQLiteLexicalSearchSettings(
            dir="kb/search/sqlite_lexical",
            filename="index.sqlite",
        ),
        enable_lexical=True,
    )
