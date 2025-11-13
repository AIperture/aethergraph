from __future__ import annotations
import os, json, pickle
from typing import List, Dict, Any
import numpy as np
try:
    import faiss
except Exception as e:
    faiss = None

from .base import VectorIndex

"""A simple FAISS index per corpus (L2 on normalized vectors ~ cosine).
Stores vectors as BLOBs along with metadata in a simple schema.
"""

class FAISSVectorIndex(VectorIndex):
    """A simple FAISS index per corpus (L2 on normalized vectors ~ cosine)."""
    def __init__(self, index_path: str, dim: int | None = None):
        super().__init__(index_path)
        self.dim = dim  # optional default; will infer on first add
        os.makedirs(index_path, exist_ok=True)

    def _paths(self, corpus_id: str):
        base = os.path.join(self.index_path, corpus_id)
        return base + ".index", base + ".meta.pkl"

    def _load(self, corpus_id: str):
        idx_path, meta_path = self._paths(corpus_id)
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            return None, []
        if faiss is None:
            raise RuntimeError("FAISS not installed")
        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            metas = pickle.load(f)
        return index, metas

    def _save(self, corpus_id: str, index, metas):
        idx_path, meta_path = self._paths(corpus_id)
        if faiss is None:
            raise RuntimeError("FAISS not installed")
        faiss.write_index(index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metas, f)

    async def add(self, corpus_id: str, chunk_ids: List[str], vectors: List[list[float]], metas: List[Dict[str,Any]]):
        if faiss is None:
            raise RuntimeError("FAISS not installed")
        vecs = np.asarray(vectors, dtype=np.float32)
        # normalize for cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
        d = vecs.shape[1]
        index, old_metas = self._load(corpus_id)
        if index is None:
            index = faiss.IndexFlatIP(d)  # cosine via normalized dot
            old_metas = []
        index.add(vecs)
        for cid, m in zip(chunk_ids, metas):
            old_metas.append({"chunk_id": cid, "meta": m})
        self._save(corpus_id, index, old_metas)

    async def delete(self, corpus_id: str, chunk_ids: List[str] | None = None):
        # Simple approach: rebuild if filtering; or delete entire corpus.
        if not chunk_ids:
            idx_path, meta_path = self._paths(corpus_id)
            for p in (idx_path, meta_path):
                if os.path.exists(p): os.remove(p)
        else:
            index, metas = self._load(corpus_id)
            if index is None:
                return
            # Rebuild without those ids
            keep = [i for i,m in enumerate(metas) if m["chunk_id"] not in set(chunk_ids)]
            if not keep:
                await self.delete(corpus_id, None)
                return
            # Need stored vectors to rebuild — this simple implementation does not persist them.
            # In production, persist vectors or recompute from text.
            raise NotImplementedError("Selective delete requires stored vectors; not implemented here.")

    async def list_chunks(self, corpus_id: str) -> List[str]:
        _, metas = self._load(corpus_id)
        return [m["chunk_id"] for m in metas] if metas else []

    async def search(self, corpus_id: str, query_vec: list[float], k: int):
        if faiss is None:
            raise RuntimeError("FAISS not installed")
        index, metas = self._load(corpus_id)
        if index is None:
            return []
        q = np.asarray([query_vec], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        D, I = index.search(q, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(metas):
                continue
            out.append({"chunk_id": metas[idx]["chunk_id"], "score": float(score), "meta": metas[idx]["meta"]})
        return out