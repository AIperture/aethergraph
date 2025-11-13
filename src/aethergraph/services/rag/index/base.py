from __future__ import annotations
from typing import List, Dict, Any

class VectorIndex:
    def __init__(self, index_path: str):
        self.index_path = index_path

    async def add(self, corpus_id: str, chunk_ids: List[str], vectors: List[list[float]], metas: List[Dict[str,Any]]):
        raise NotImplementedError

    async def delete(self, corpus_id: str, chunk_ids: List[str] | None = None):
        raise NotImplementedError

    async def search(self, corpus_id: str, query_vec: list[float], k: int) -> List[Dict[str,Any]]:
        """Return a list of {{chunk_id, score, meta}} sorted by descending score."""
        raise NotImplementedError

    async def list_chunks(self, corpus_id: str) -> List[str]:
        raise NotImplementedError