from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import hmac
import os
import threading
from typing import Any

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.services.continuations.continuation import Continuation, Correlator


class InMemoryContinuationStore(AsyncContinuationStore):
    """
    Process-local, in-memory continuation store.

    - Thread-safe via RLock (main loop + sidecar can share it).
    - No persistence beyond process lifetime.

    NOTE: suitable for testing or single-process deployments only.
    """

    def __init__(self, *, secret: bytes):
        self._secret = secret
        self._by_id: dict[tuple[str, str], Continuation] = {}
        self._by_token: dict[str, tuple[str, str]] = {}
        self._corr_tokens: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    # --- helpers ---

    def _key(self, run_id: str, node_id: str) -> tuple[str, str]:
        return (run_id, node_id)

    def _corr_key(self, corr: Correlator) -> str:
        scheme, channel, thread, message = corr.key()
        return f"{scheme}:{channel}:{thread}:{message}"

    def _hmac(self, *parts: str) -> str:
        h = hmac.new(self._secret, digestmod=hashlib.sha256)
        for p in parts:
            h.update(p.encode("utf-8"))
        return h.hexdigest()

    async def mint_token(self, run_id: str, node_id: str, attempts: int) -> str:
        return self._hmac(run_id, node_id, str(attempts), os.urandom(8).hex())

    # --- core ---

    async def save(self, cont: Continuation) -> None:
        with self._lock:
            self._by_id[self._key(cont.run_id, cont.node_id)] = cont
            self._by_token[cont.token] = (cont.run_id, cont.node_id)

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        with self._lock:
            return self._by_id.get(self._key(run_id, node_id))

    async def delete(self, run_id: str, node_id: str) -> None:
        with self._lock:
            key = self._key(run_id, node_id)
            cont = self._by_id.pop(key, None)
            if cont:
                self._by_token.pop(cont.token, None)

    async def list_cont_by_run(self, run_id: str) -> list[Continuation]:
        with self._lock:
            return [c for (r, _), c in self._by_id.items() if r == run_id]

    # --- token ---

    async def get_by_token(self, token: str) -> Continuation | None:
        with self._lock:
            ref = self._by_token.get(token)
            if not ref:
                return None
            run_id, node_id = ref
            return self._by_id.get(self._key(run_id, node_id))

    async def mark_closed(self, token: str) -> None:
        c = await self.get_by_token(token)
        if not c:
            return
        if not c.closed:
            c.closed = True
            await self.save(c)

    async def verify_token(self, run_id: str, node_id: str, token: str) -> bool:
        c = await self.get(run_id, node_id)
        return bool(c and hmac.compare_digest(token, c.token))

    # --- correlators ---

    async def bind_correlator(self, *, token: str, corr: Correlator) -> None:
        key = self._corr_key(corr)
        with self._lock:
            toks = self._corr_tokens.setdefault(key, [])
            if token not in toks:
                toks.append(token)

    async def find_by_correlator(self, *, corr: Correlator) -> Continuation | None:
        key = self._corr_key(corr)
        with self._lock:
            toks = list(self._corr_tokens.get(key, []))

        now = datetime.now(timezone.utc)
        for tok in reversed(toks):
            c = await self.get_by_token(tok)
            if not c or c.closed:
                continue
            if c.deadline and now > c.deadline.astimezone(timezone.utc):
                continue
            return c
        return None

    # --- scans ---

    async def last_open(self, *, channel: str, kind: str) -> Continuation | None:
        with self._lock:
            candidates = [
                c
                for c in self._by_id.values()
                if not c.closed and c.channel == channel and c.kind == kind
            ]
        if not candidates:
            return None

        # pick most recent created_at
        def _ts(c: Continuation) -> float:
            dt = c.created_at or datetime.min.replace(tzinfo=timezone.utc)
            return dt.timestamp()

        return max(candidates, key=_ts)

    async def list_waits(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(c) for c in self._by_id.values()]

    async def clear(self) -> None:
        with self._lock:
            self._by_id.clear()
            self._by_token.clear()
            self._corr_tokens.clear()

    async def alias_for(self, token: str) -> str | None:
        return token[:24]
