from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
import pickle
import threading
from typing import Any

import numpy as np

from aethergraph.contracts.storage.vector_index import VectorIndex

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


class FAISSVectorIndex(VectorIndex):
    """
    Simple FAISS-backed index, one .index + .meta.pkl per corpus_id.
    Uses cosine similarity via normalized vectors (IndexFlatIP).
    """

    def __init__(self, root: str, dim: int | None = None):
        if faiss is None:
            raise RuntimeError("FAISSVectorIndex requires `faiss` to be installed.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim  # optional default dimension
        self._lock = threading.RLock()

    def _paths(self, corpus_id: str) -> tuple[Path, Path]:
        base = self.root / corpus_id
        return base.with_suffix(".index"), base.with_suffix(".meta.pkl")

    def _load_sync(self, corpus_id: str):
        idx_path, meta_path = self._paths(corpus_id)
        if not (idx_path.exists() and meta_path.exists()):
            return None, []
        with self._lock:
            index = faiss.read_index(str(idx_path))
            with meta_path.open("rb") as f:
                metas = pickle.load(f)
        return index, metas

    def _save_sync(self, corpus_id: str, index, metas: list[dict[str, Any]]) -> None:
        idx_path, meta_path = self._paths(corpus_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(index, str(idx_path))
            with meta_path.open("wb") as f:
                pickle.dump(metas, f)

    async def add(
        self,
        corpus_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        metas: list[dict[str, Any]],
    ) -> None:
        if faiss is None:
            raise RuntimeError("FAISS not installed")
        if not chunk_ids:
            return

        vecs = np.asarray(vectors, dtype=np.float32)
        # Normalize for cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms

        def _add_sync():
            index, old_metas = self._load_sync(corpus_id)
            d = vecs.shape[1]
            if index is None:
                # If dim was provided, optionally sanity-check
                if self.dim is not None and self.dim != d:
                    raise ValueError(f"FAISSVectorIndex: dim mismatch {self.dim} vs {d}")
                index = faiss.IndexFlatIP(d)
                old_metas = []
                self.dim = d

            index.add(vecs)
            for cid, m in zip(chunk_ids, metas, strict=True):
                old_metas.append({"chunk_id": cid, "meta": m})
            self._save_sync(corpus_id, index, old_metas)

        await asyncio.to_thread(_add_sync)

    async def delete(self, corpus_id: str, chunk_ids: list[str] | None = None) -> None:
        if chunk_ids is None:
            # Delete entire corpus
            idx_path, meta_path = self._paths(corpus_id)
            for p in (idx_path, meta_path):
                with suppress(Exception):
                    p.unlink(missing_ok=True)
            return

        # Selective delete is tricky with plain FAISS; we’d need to rebuild.
        # For now, keep explicit about limitations:
        async def _delete_sync():
            index, metas = self._load_sync(corpus_id)
            if index is None:
                return
            keep_idxs = [i for i, m in enumerate(metas) if m["chunk_id"] not in set(chunk_ids)]
            if not keep_idxs:
                # Remove whole corpus
                idx_path, meta_path = self._paths(corpus_id)
                for p in (idx_path, meta_path):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception as e:
                        import logging

                        logger = logging.getLogger("aethergraph.storage.vector_index.faiss_index")
                        logger.error(f"Failed to delete {p}: {e}")
                return

            # Rebuild index with kept vectors
            # NOTE: we do NOT have original vectors here,
            # so in a minimal implementation we simply raise.
            raise NotImplementedError(
                "FAISSVectorIndex: selective delete requires storing vectors; "
                "either extend metadata to keep vectors, or rebuild from source."
            )

        await asyncio.to_thread(_delete_sync)

    async def list_chunks(self, corpus_id: str) -> list[str]:
        def _list_sync() -> list[str]:
            _, metas = self._load_sync(corpus_id)
            return [m["chunk_id"] for m in metas] if metas else []

        return await asyncio.to_thread(_list_sync)

    async def list_corpora(self) -> list[str]:
        def _scan() -> list[str]:
            out = []
            for p in self.root.glob("*.meta.pkl"):
                out.append(p.stem)  # strip .meta.pkl => corpus_id
            return out

        return await asyncio.to_thread(_scan)

    async def search(
        self,
        corpus_id: str,
        query_vec: list[float],
        k: int,
        where: dict[str, Any] | None = None,
        max_candidates: int | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        FAISS-backed search with compatibility to SQLiteVectorIndex:

        - where: equality filters on metadata (e.g., org_id, user_id, scope_id, etc.)
        - created_at_min / created_at_max: numeric UNIX timestamps for time-range filtering.
        - max_candidates: how many FAISS hits to retrieve before filtering.

        Since FAISS doesn't support filtering natively, we:
          1) Search across all vectors (or up to max_candidates).
          2) Manually filter results by `where` and time bounds.
        """

        if faiss is None:
            raise RuntimeError("FAISS not installed")

        where = where or {}

        # Normalize query vector for cosine similarity
        q = np.asarray([query_vec], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        def _search_sync() -> list[dict[str, Any]]:
            index, metas = self._load_sync(corpus_id)
            if index is None or not metas:
                return []

            n = len(metas)
            if n == 0:
                return []

            # How many neighbors to ask FAISS for:
            # - k here is "raw_k" from SearchBackend (e.g., top_k * 3)
            # - max_candidates is an outer cap (e.g., top_k * 50)
            search_k = min(
                n,
                max_candidates or n,
            )
            if search_k <= 0:
                return []

            # Ask FAISS for the top search_k neighbors
            D, I = index.search(q, search_k)  # noqa: E741
            scores = D[0].tolist()
            idxs = I[0].tolist()

            out: list[dict[str, Any]] = []

            for score, idx in zip(scores, idxs, strict=True):
                if idx < 0 or idx >= len(metas):
                    continue

                m = metas[idx]  # {"chunk_id": ..., "meta": {...}}
                meta = dict(m.get("meta") or {})

                # --- Apply `where` equality filters ----------------------
                match = True
                for key, val in where.items():
                    if val is None:
                        continue
                    if meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue

                # --- Apply time-window filters ---------------------------
                cat = meta.get("created_at_ts")
                # If we have a time bound but no created_at_ts, we treat as non-match
                if created_at_min is not None and (
                    cat is None or float(cat) < float(created_at_min)
                ):
                    continue
                if created_at_max is not None and (
                    cat is None or float(cat) > float(created_at_max)
                ):
                    continue

                out.append(
                    {
                        "chunk_id": m["chunk_id"],
                        "score": float(score),
                        "meta": meta,
                    }
                )

                # Stop once we've collected k matches
                if len(out) >= k:
                    break

            return out

        return await asyncio.to_thread(_search_sync)
