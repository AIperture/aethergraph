from __future__ import annotations

from typing import Any


class VectorIndex:
    def __init__(self, index_path: str):
        self.index_path = index_path

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metas: list[dict[str, Any]],
    ):
        raise NotImplementedError

    async def delete(self, corpus_id: str, chunk_ids: list[str] | None = None):
        raise NotImplementedError

    async def search(self, corpus_id: str, query_vec: list[float], k: int) -> list[dict[str, Any]]:
        """Return a list of {{chunk_id, score, meta}} sorted by descending score."""
        raise NotImplementedError

    async def list_chunks(self, corpus_id: str) -> list[str]:
        raise NotImplementedError
