from __future__ import annotations

import asyncio
from typing import Any

from aethergraph.contracts.storage.vector_index import VectorIndex

try:
    import chromadb
except Exception:
    chromadb = None

"""Adapter over a chromadb.Client.
NOTE: this is just a stub implementation to show how vector indexes can be adapted.
We can use the same VectorIndex interface for different backends with adaptation.

e.g.
- QdrantVectorIndex: corpus_id -> Qdrant collection, chunk_id -> point ID, vector + metadata stored per point.
- PineconeVectorIndex: corpus_id -> Pinecone index, chunk_id -> vector ID, vector + metadata stored per vector.
- WeaviateVectorIndex: corpus_id -> Weaviate class, chunk_id -> object ID, vector + metadata stored per object.
"""


class ChromaVectorIndex(VectorIndex):
    """
    Adapter over a chromadb.Client.
    Each `corpus_id` is mapped to a Chroma collection:
        collection_name = f"{collection_prefix}{corpus_id}"
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        *,
        collection_prefix: str = "rag_",
    ):
        if chromadb is None:
            raise RuntimeError("ChromaVectorIndex requires `chromadb` to be installed.")
        self._client = client
        self._prefix = collection_prefix

    def _collection_name(self, corpus_id: str) -> str:
        return f"{self._prefix}{corpus_id}"

    def _collection(self, corpus_id: str):
        name = self._collection_name(corpus_id)
        return self._client.get_or_create_collection(name=name)

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metas: list[dict[str, Any]],
    ) -> None:
        if not chunk_ids:
            return
        col = self._collection(corpus_id)
        # Chroma's API is sync, but fast; you can wrap in to_thread if needed.
        col.upsert(
            ids=chunk_ids,  # type: IDs
            embeddings=vectors,  # type: Embeddings
            metadatas=metas,  # type: Metadatas
        )

    async def delete(self, corpus_id: str, chunk_ids: list[str] | None = None) -> None:
        name = self._collection_name(corpus_id)
        if chunk_ids is None:
            # Drop whole collection
            try:
                self._client.delete_collection(name=name)
            except Exception as err:
                raise RuntimeError(f"Failed to delete collection '{name}'") from err
            return

        try:
            col = self._client.get_collection(name=name)
        except Exception:
            return
        col.delete(ids=chunk_ids)

    async def search(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        try:
            col = self._collection(corpus_id)
        except Exception:
            return []

        def _search_sync() -> list[dict[str, Any]]:
            # Chroma's client is sync; do this in a thread so we don't block the loop.
            res = col.query(
                query_embeddings=[query_vec],
                n_results=k,
            )

            # Chroma returns lists-of-lists
            ids = (res.get("ids") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]

            out: list[dict[str, Any]] = []
            for cid, dist, meta in zip(ids, dists, metas, strict=True):
                # Chroma distance is "smaller is better".
                # We convert to a "score" where larger is better.
                # Any monotone transform is fine; here we use negative distance.
                score = float(-dist)
                out.append(
                    {
                        "chunk_id": cid,
                        "score": score,
                        "meta": meta or {},
                    }
                )
            return out

        return await asyncio.to_thread(_search_sync)

    async def list_corpora(self) -> list[str]:
        cols = self._client.list_collections()
        out = []
        for c in cols:
            name = c.name
            if name.startswith(self._prefix):
                out.append(name[len(self._prefix) :])
        return out

    async def list_chunks(self, corpus_id: str) -> list[str]:
        try:
            col = self._collection(corpus_id)
        except Exception:
            return []
        # Chroma `get()` can be expensive for huge collections; fine for small/mid-scale.
        res = col.get()
        return list(res.get("ids") or [])
