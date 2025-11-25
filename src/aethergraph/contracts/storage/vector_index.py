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
    ) -> list[dict[str, Any]]: ...

    # Each dict MUST look like:
    # {"chunk_id": str, "score": float, "meta": dict[str, Any]}

    # Optional
    async def list_corpora(self) -> list[str]: ...
    async def list_chunks(self, corpus_id: str) -> list[str]: ...
