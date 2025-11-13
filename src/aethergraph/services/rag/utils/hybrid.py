from __future__ import annotations
import math, re
from collections import Counter
from typing import List, Dict, Any

def lexical_score(query: str, text: str) -> float:
    # Extremely lightweight bag-of-words match score.
    ws = lambda s: re.findall(r"\w+", s.lower())
    q = ws(query)
    t = ws(text)
    if not q or not t:
        return 0.0
    cq = Counter(q); ct = Counter(t)
    # normalized term overlap
    overlap = sum(min(cq[w], ct.get(w, 0)) for w in cq)
    return overlap / (sum(cq.values()) + 1e-9)

def fuse_scores(dense_score: float, lexical: float, alpha: float = 0.8) -> float:
    # Linear fusion; alpha favors dense similarity.
    return alpha * dense_score + (1.0 - alpha) * lexical

def topk_fuse(query: str, dense_hits: List[Dict[str,Any]], chunk_lookup: Dict[str, str], k: int) -> List[Dict[str,Any]]:
    out = []
    for h in dense_hits:
        txt = chunk_lookup.get(h["chunk_id"], "")
        lex = lexical_score(query, txt)
        fused = fuse_scores(h.get("score", 0.0), lex)
        out.append({**h, "score": fused})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:k]