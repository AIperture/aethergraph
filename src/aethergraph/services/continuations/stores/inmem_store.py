from __future__ import annotations
import os, hmac, hashlib, threading
from datetime import datetime, timezone
from typing import Dict, Optional, List
from collections import defaultdict

from ..continuation import Continuation, Correlator

class InMemoryContinuationStore:  # implements AsyncContinuationStore
    def __init__(self, secret: bytes | None = None):
        self.secret = secret or os.urandom(32)
        self._by_token: Dict[str, Continuation] = {}
        self._by_run_node: Dict[tuple[str,str], Continuation] = {}
        self._corr_index: Dict[tuple[str,str,str,str], List[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def _hmac(self, *parts: str) -> str:
        h = hmac.new(self.secret, digestmod=hashlib.sha256)
        for part in parts: h.update(part.encode("utf-8"))
        return h.hexdigest()

    async def mint_token(self, run_id: str, node_id: str, attempts: int) -> str:
        return self._hmac(run_id, node_id, str(attempts), os.urandom(8).hex())

    async def save(self, cont: Continuation) -> None:
        with self._lock:
            self._by_token[cont.token] = cont
            self._by_run_node[(cont.run_id, cont.node_id)] = cont

    async def get(self, run_id: str, node_id: str) -> Optional[Continuation]:
        with self._lock:
            return self._by_run_node.get((run_id, node_id))

    async def delete(self, run_id: str, node_id: str) -> None:
        with self._lock:
            c = self._by_run_node.pop((run_id, node_id), None)
            if c: self._by_token.pop(c.token, None)

    async def get_by_token(self, token: str) -> Optional[Continuation]:
        with self._lock:
            return self._by_token.get(token)

    async def mark_closed(self, token: str) -> None:
        with self._lock:
            c = self._by_token.get(token)
            if c: c.closed = True

    async def verify_token(self, run_id: str, node_id: str, token: str) -> bool:
        c = await self.get(run_id, node_id)
        return bool(c and hmac.compare_digest(c.token, token))

    async def bind_correlator(self, *, token: str, corr: Correlator) -> None:
        key = corr.key()
        with self._lock:
            toks = self._corr_index[key]
            if token not in toks:
                toks.append(token)

    async def find_by_correlator(self, *, corr: Correlator) -> Optional[Continuation]:
        with self._lock:
            toks = list(self._corr_index.get(corr.key()) or [])
        for t in reversed(toks):
            c = await self.get_by_token(t)
            if c and not c.closed:
                if c.deadline and datetime.now(timezone.utc) > c.deadline.astimezone(timezone.utc):
                    continue
                return c
        return None

    async def last_open(self, *, channel: str, kind: str) -> Optional[Continuation]:
        with self._lock:
            for c in reversed(list(self._by_token.values())):
                if not c.closed and c.channel == channel and c.kind == kind:
                    return c
        return None

    async def list_waits(self) -> list[dict]:
        with self._lock:
            return [c.to_dict() for c in self._by_token.values() if not c.closed]

    async def clear(self) -> None:
        with self._lock:
            self._by_token.clear(); self._by_run_node.clear(); self._corr_index.clear()

    async def alias_for(self, token: str) -> Optional[str]:
        return token[:24]
