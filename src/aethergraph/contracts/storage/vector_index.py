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
    ) -> None: ...

    async def delete(
        self,
        corpus_id: str,
        chunk_ids: list[str] | None = None,
    ) -> None: ...

    async def search(
        self, corpus_id: str, query_vec: list[float], k: int
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    # Optional
    async def list_corpora(self) -> list[str]: ...
    async def list_chunks(self, corpus_id: str) -> list[str]: ...
