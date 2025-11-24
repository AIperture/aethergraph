from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import hmac
import os
from typing import Any

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.kv import AsyncKV
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.continuations.continuation import Continuation, Correlator


class KVDocContinuationStore(AsyncContinuationStore):
    """
    Continuation store backed by:
      - DocStore: main continuation document (one per (run_id, node_id))
      - AsyncKV:  token and correlator indices
      - EventLog: (optional) audit trail of continuation events
    """

    def __init__(
        self,
        *,
        doc_store: DocStore,
        kv: AsyncKV,
        event_log: EventLog | None = None,
        secret: bytes,
        namespace: str = "cont",
    ):
        self._docs = doc_store
        self._kv = kv
        self._log = event_log
        self._secret = secret
        self._ns = namespace.rstrip("/")  # namespace prefix for KV keys

    # ---------- key helpers ----------
    def _cont_id(self, run_id: str, node_id: str) -> str:
        # one doc per continuation
        return f"{self._ns}/runs/{run_id}/nodes/{node_id}"

    def _token_key(self, token: str) -> str:
        return f"{self._ns}:token:{token}"

    def _corr_key(self, corr: Correlator) -> str:
        scheme, channel, thread, message = corr.key()
        return f"{self._ns}:corr:{scheme}:{channel}:{thread}:{message}"

    # ---------- token helpers ----------
    def _hmac(self, *parts: str) -> str:
        h = hmac.new(self._secret, digestmod=hashlib.sha256)
        for p in parts:
            h.update(p.encode("utf-8"))
        return h.hexdigest()

    async def mint_token(self, run_id: str, node_id: str, attempts: int) -> str:
        token = self._hmac(run_id, node_id, str(attempts), os.urandom(8).hex())
        return token

    # ---------- main methods ----------
    async def save(self, cont: Continuation) -> None:
        payload = cont.to_dict() if hasattr(cont, "to_dict") else asdict(cont)

        # Normalize datetime fields to ISO format
        for k in ("deadline", "next_wakeup_at", "created_at"):
            v = payload.get(k)
            if isinstance(v, datetime):
                payload[k] = v.astimezone(timezone.utc).isoformat()

        doc_id = self._cont_id(cont.run_id, cont.node_id)
        await self._docs.put(doc_id, payload)

        # token -> (run_id, node_id)
        await self._kv.set(
            self._token_key(cont.token),
            {"run_id": cont.run_id, "node_id": cont.node_id},
        )

        if self._log is not None:
            evt = {
                "scope_id": cont.run_id,
                "kind": "continuation.save",
                "ts": datetime.now(timezone.utc).isoformat(),
                "tags": [cont.channel or "", cont.kind or ""],
                "payload": payload,
            }
            await self._log.append(evt)

    async def _doc_to_cont(self, data: dict[str, Any] | None) -> Continuation | None:
        if data is None:
            return None

        for k in ("deadline", "next_wakeup_at", "created_at"):
            v = data.get(k)
            if isinstance(v, str):
                data[k] = datetime.fromisoformat(v)

        data["closed"] = bool(data.get("closed", False))
        return Continuation(**data)

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        doc_id = self._cont_id(run_id, node_id)
        data = await self._docs.get(doc_id)
        return await self._doc_to_cont(data)

    async def list_cont_by_runs(self, run_id: str) -> list[Continuation]:
        prefix = f"{self._ns}/runs/{run_id}/nodes/"
        ids = await self._docs.list()
        out: list[Continuation] = []
        for doc_id in ids:
            if not doc_id.startswith(prefix):
                continue
            data = await self._docs.get(doc_id)
            cont = await self._doc_to_cont(data)
            if cont is not None:
                out.append(cont)
        return out

    async def delete(self, run_id: str, node_id: str) -> None:
        cont = await self.get(run_id, node_id)
        doc_id = self._cont_id(run_id, node_id)
        await self._docs.delete(doc_id)
        if cont:
            # best-effort delete of token mapping
            await self._kv.delete(self._token_key(cont.token))

    # ---------- token methods ----------
    async def get_by_token(self, token: str) -> Continuation | None:
        ref = await self._kv.get(self._token_key(token), default=None)
        if not ref:
            return None
        run_id = ref.get("run_id")
        node_id = ref.get("node_id")
        if not run_id or not node_id:
            return None
        return await self.get(run_id, node_id)

    async def mark_closed(self, token: str) -> None:
        cont = await self.get_by_token(token)
        if not cont:
            return
        if not cont.closed:
            cont.closed = True
            await self.save(cont)

    async def verify_token(self, run_id: str, node_id: str, token: str) -> bool:
        cont = await self.get(run_id, node_id)
        return bool(cont and hmac.compare_digest(token, cont.token))

    # ---------- correlator methods ----------
    async def bind_correlator(self, *, token: str, corr: str) -> None:
        key = self._corr_key(corr)
        toks: list[str] = await self._kv.get(key, default=[]) or []
        if token not in toks:
            toks.append(token)
            await self._kv.set(key, toks)

        # TODO: consider reverse mapping: token -> correlators for easier cleanup
        # for now we don't need that

    async def find_by_correlator(self, *, corr: Correlator) -> list[Continuation] | None:
        """
        Find all continuations matching the correlator. Following the method:
        - get all tokens for the correlator
        - fetch each continuation by token
        - filter out closed or expired continuations
        - return the first valid continuation found (or None)
        """
        key = self._corr_key(corr)
        toks: list[str] = await self._kv.get(key, default=[]) or []
        from datetime import datetime as dt, timezone as tz

        for tok in reversed(toks):
            cont = await self.get_by_token(tok)
            if not cont or cont.closed:
                continue
            if cont.deadline and dt.now(tz.utc) > cont.deadline.astimezone(tz.utc):
                continue
            return cont
        return None

    # ---------- scans ----------
    async def last_open(self, *, channel: str, kind: str) -> Continuation | None:
        """
        Slow scan (like FS version) – OK for dev scale.
        We scan all docs and pick the most recent created_at with matching channel/kind.
        NOTE: this is only for debugging and development purposes. Do not use in production!
        """
        ids = await self._docs.list()
        best: Continuation | None = None
        best_ts: float | None = None

        for doc_id in ids:
            if not doc_id.startswith(f"{self._ns}/runs/"):
                continue
            data = await self._docs.get(doc_id)
            cont = await self._doc_to_cont(data)
            if not cont or cont.closed:
                continue
            if cont.channel != channel or cont.kind != kind:
                continue
            created = cont.created_at or datetime.min.replace(tzinf=timezone.utc)
            ts = created.timestamp()
            if best_ts is None or ts > best_ts:
                best = cont
                best_ts = ts
        return best

    async def list_waits(self) -> list[dict[str, Any]]:
        """
        Return all continuations as dicts (similar to FS version).
        Caller can filter for waits if needed.
        """
        ids = await self._docs.list()
        out: list[dict[str, Any]] = []
        for doc_id in ids:
            if not doc_id.startswith(f"{self._ns}/runs/"):
                continue
            data = await self._docs.get(doc_id)
            if data:
                out.append(data)
        return out

    async def clear(self) -> None:
        """
        Best-effort clear of continuation documents.

        NOTE: KV indexes may remain unless we have a scan-capable KV implementation.
        They are harmless: get_by_token() will just return None if the doc is gone.
        """
        # 1) DocStore cleanup
        ids = await self._docs.list()
        for doc_id in ids:
            if doc_id.startswith(f"{self._ns}/runs/"):
                await self._docs.delete(doc_id)

        # 2) Best-effort KV cleanup if scan_keys is available
        # TODO: implement KV indexes if we have scan capability
        scan = getattr(self._kv, "scan_keys", None)
        if callable(scan):
            token_prefix = f"{self._ns}:token:"
            corr_prefix = f"{self._ns}:corr:"

            for pfx in (token_prefix, corr_prefix):
                try:
                    keys: list[str] = await scan(pfx)  # type: ignore[call-arg]
                except Exception:
                    # If a backend throws here, we still consider clear() successful
                    continue
                for k in keys:
                    try:
                        await self._kv.delete(k)
                    except Exception:
                        # swallow individual key errors; we're best-effort here
                        continue

    async def alias_for(self, token: str) -> str | None:
        return token[:24]
