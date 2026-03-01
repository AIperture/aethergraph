from __future__ import annotations

from collections import Counter
import re
from typing import Any


def lexical_score(query: str, text: str) -> float:
    """Extremely lightweight bag-of-words match score."""

    def ws(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower())

    q = ws(query)
    t = ws(text)
    if not q or not t:
        return 0.0

    cq = Counter(q)
    ct = Counter(t)
    overlap = sum(min(cq[w], ct.get(w, 0)) for w in cq)
    return overlap / (sum(cq.values()) + 1e-9)


def fuse_scores(dense_score: float, lexical: float, alpha: float = 0.8) -> float:
    """Linear fusion; alpha favors dense similarity."""
    return alpha * dense_score + (1.0 - alpha) * lexical


def rerank_hybrid(
    *,
    query: str,
    hits: list[dict[str, Any]],
    chunk_lookup: dict[str, str],
    alpha: float = 0.8,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Given dense hits (with 'chunk_id' and 'score') + a chunk_id->text map,
    recompute a fused dense+lexical score and return reranked hits.
    """
    out: list[dict[str, Any]] = []
    for h in hits:
        cid = h.get("chunk_id")
        txt = chunk_lookup.get(cid or "", "")
        lex = lexical_score(query, txt)
        fused = fuse_scores(h.get("score", 0.0), lex, alpha=alpha)
        out.append({**h, "score": fused})

    out.sort(key=lambda x: x["score"], reverse=True)

    if top_k is not None:
        return out[:top_k]
    return out
