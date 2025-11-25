import logging
from pathlib import Path

from aethergraph.utils.optdeps import require

logger = logging.getLogger("aethergraph.rag.index_factory")


def _default_index_path(root: str, backend: str) -> str:
    base = Path(root) / "rag_index"
    if backend == "faiss":
        return str(base / "faiss.index")
    return str(base / "sqlite.index")


def create_vector_index(
    *, backend: str, index_path: str | None, dim: int | None, root: str = "./aethergraph_data/rag"
):
    """
    Create a vector index instance. Supported backends: 'sqlite', 'faiss'.
    Falls back to 'sqlite' if FAISS is unavailable.
    """
    backend = (backend or "sqlite").lower()
    if backend not in {"sqlite", "faiss"}:
        logger.warning(f"Unknown RAG backend {backend!r}; falling back to sqlite.")
        backend = "sqlite"

    if backend == "faiss":
        # try FAISS, fallback to sqlite with a warning
        try:
            require("faiss", "faiss")  # faiss-cpu exposes module 'faiss'
            from .index.faiss_index import FAISSVectorIndex

            path = (
                str(Path(index_path) / "faiss")
                if index_path is not None
                else _default_index_path(root, "faiss")
            )
            return FAISSVectorIndex(path, dim=dim)
        except Exception as e:
            logger.warning(f"FAISS backend unavailable ({e}); falling back to sqlite.")
            backend = "sqlite"

    # sqlite (default)
    from aethergraph.storage.vector_index.sqlite_index import SQLiteVectorIndex

    path = (
        str(Path(index_path) / "sqlite")
        if index_path is not None
        else _default_index_path(root, "sqlite")
    )
    return SQLiteVectorIndex(path)
