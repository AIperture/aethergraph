"""Legacy-active search settings isolated pending the S9 canonical cut."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SQLiteLexicalSearchSettings(BaseModel):
    """
    Settings for SQLite-based lexical search backend.
    Paths are relative to AppSettings.workspace.
    """

    dir: str = "search/sqlite_lexical"
    filename: str = "index.sqlite"


class SQLiteSearchVectorSettings(BaseModel):
    """Legacy-active SQLite vector settings owned only by search composition."""

    dir: str = "search/vector_sqlite"


class FAISSSearchVectorSettings(BaseModel):
    """Legacy-active FAISS vector settings owned only by search composition."""

    dir: str = "search/vector_faiss"
    dim: int | None = None


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

    sqlite_vector: SQLiteSearchVectorSettings = SQLiteSearchVectorSettings()
    faiss_vector: FAISSSearchVectorSettings = FAISSSearchVectorSettings()

    # Lexical search backend (for pure sqlite_lexical backend, *and* for
    # optional lexical index when enable_lexical=True on vector backends).
    sqlite_lexical: SQLiteLexicalSearchSettings = SQLiteLexicalSearchSettings()

    # NEW: toggle lexical index when using vector backends
    enable_lexical: bool = True
