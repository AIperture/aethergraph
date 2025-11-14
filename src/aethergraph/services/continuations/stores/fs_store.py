from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any

from ..continuation import Continuation, Correlator


class FSContinuationStore:  # implements AsyncContinuationStore
    def __init__(self, root: str | Path, secret: bytes):
        self.root = Path(root)
        self.secret = secret
        self._lock = threading.RLock()

    # ---------- helpers ----------
    def _cont_path(self, run_id: str, node_id: str) -> Path:
        return self.root / "runs" / run_id / "nodes" / node_id / "continuation.json"

    def _token_idx_path(self, token: str) -> Path:
        return self.root / "index" / "tokens" / f"{token}.json"

    def _rev_idx_path(self, token: str) -> Path:
        return self.root / "index" / "rev" / f"{token}.json"

    def _safe(self, s: str) -> str:
        s = (s or "").replace("\\", "_").replace("/", "_").replace(":", "_")
        s = re.sub(r"[^A-Za-z0-9._@-]", "_", s)
        s = re.sub(r"_+", "_", s).strip("._ ") or "x"
        if len(s) > 100:
            h = hashlib.sha1(s.encode()).hexdigest()[:8]
            s = f"{s[:92]}_{h}"
        return s

    def _corr_dir(self, scheme: str, channel: str, thread: str, message: str) -> Path:
        return (
            self.root
            / "index"
            / "corr"
            / self._safe(scheme)
            / self._safe(channel)
            / self._safe(thread)
            / self._safe(message)
        )

    def _corr_tokens_path(self, scheme: str, channel: str, thread: str, message: str) -> Path:
        return self._corr_dir(scheme, channel, thread, message) / "tokens.json"

    def _hmac(self, *parts: str) -> str:
        h = hmac.new(self.secret, digestmod=hashlib.sha256)
        for p in parts:
            h.update(p.encode("utf-8"))
        return h.hexdigest()

    async def mint_token(self, run_id: str, node_id: str, attempts: int) -> str:
        return self._hmac(run_id, node_id, str(attempts), os.urandom(8).hex())

    # ---------- core ----------
    async def save(self, cont: Continuation) -> None:
        def _write():
            path = self._cont_path(cont.run_id, cont.node_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = cont.to_dict() if hasattr(cont, "to_dict") else asdict(cont)
            for k in ("deadline", "next_wakeup_at", "created_at"):
                v = payload.get(k)
                if isinstance(v, datetime):
                    payload[k] = v.astimezone(timezone.utc).isoformat()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
            self._write_token_index(cont.run_id, cont.node_id, cont.token)

        await asyncio.to_thread(_write)

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        def _read():
            path = self._cont_path(run_id, node_id)
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            for k in ("deadline", "next_wakeup_at", "created_at"):
                if raw.get(k):
                    raw[k] = datetime.fromisoformat(raw[k])
            raw["closed"] = bool(raw.get("closed", False))
            return Continuation(**raw)

        return await asyncio.to_thread(_read)

    async def delete(self, run_id: str, node_id: str) -> None:
        def _del():
            p = self._cont_path(run_id, node_id)
            if p.exists():
                p.unlink()

        await asyncio.to_thread(_del)

    # ---------- token helpers ----------
    def _write_token_index(self, run_id: str, node_id: str, token: str) -> None:
        with self._lock:
            p = self._token_idx_path(token)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({"run_id": run_id, "node_id": node_id}, indent=2), encoding="utf-8"
            )

    async def get_by_token(self, token: str) -> Continuation | None:
        def _lookup():
            p = self._token_idx_path(token)
            if not p.exists():
                return None
            ref = json.loads(p.read_text(encoding="utf-8"))
            return ref["run_id"], ref["node_id"]

        ref = await asyncio.to_thread(_lookup)
        if not ref:
            return None
        run_id, node_id = ref
        return await self.get(run_id, node_id)

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

    # ---------- correlators ----------
    async def bind_correlator(self, *, token: str, corr: Correlator) -> None:
        def _bind():
            scheme, channel, thread, message = corr.key()
            tokens_path = self._corr_tokens_path(scheme, channel, thread, message)
            tokens_path.parent.mkdir(parents=True, exist_ok=True)
            toks: list[str] = []
            if tokens_path.exists():
                try:
                    toks = json.loads(tokens_path.read_text(encoding="utf-8"))
                except Exception:
                    toks = []
            if token not in toks:
                toks.append(token)
                tokens_path.write_text(json.dumps(toks, indent=2), encoding="utf-8")
                # reverse index
                r = self._rev_idx_path(token)
                r.parent.mkdir(parents=True, exist_ok=True)
                paths = []
                if r.exists():
                    try:
                        paths = json.loads(r.read_text(encoding="utf-8"))
                    except Exception:
                        paths = []
                key_path = str(tokens_path.relative_to(self.root))
                if key_path not in paths:
                    paths.append(key_path)
                    r.write_text(json.dumps(paths, indent=2), encoding="utf-8")

        await asyncio.to_thread(_bind)

    async def find_by_correlator(self, *, corr: Correlator) -> Continuation | None:
        def _read_toks():
            scheme, channel, thread, message = corr.key()
            p = self._corr_tokens_path(scheme, channel, thread, message)
            if not p.exists():
                return []
            try:
                return json.loads(p.read_text(encoding="utf-8")) or []
            except Exception:
                return []

        toks = await asyncio.to_thread(_read_toks)
        for tok in reversed(toks):
            c = await self.get_by_token(tok)
            if c and not c.closed:
                if c.deadline and datetime.now(timezone.utc) > c.deadline.astimezone(timezone.utc):
                    continue
                return c
        return None

    async def last_open(self, *, channel: str, kind: str) -> Continuation | None:
        # Optional slow scan (dev only)
        def _scan():
            waits = []
            runs_path = self.root / "runs"
            if not runs_path.exists():
                return waits
            for run_dir in runs_path.iterdir():
                nodes_dir = run_dir / "nodes"
                if not nodes_dir.exists():
                    continue
                for node_dir in nodes_dir.iterdir():
                    cont_path = node_dir / "continuation.json"
                    if cont_path.exists():
                        waits.append(json.loads(cont_path.read_text(encoding="utf-8")))
            return waits

        waits = await asyncio.to_thread(_scan)
        for raw in reversed(waits):
            if raw.get("closed"):
                continue
            if raw.get("channel") == channel and raw.get("kind") == kind:
                for k in ("deadline", "next_wakeup_at", "created_at"):
                    if raw.get(k):
                        raw[k] = datetime.fromisoformat(raw[k])
                return Continuation(**raw)
        return None

    async def list_waits(self) -> list[dict[str, Any]]:
        def _scan():
            out = []
            runs_path = self.root / "runs"
            if not runs_path.exists():
                return out
            for run_dir in runs_path.iterdir():
                nodes_dir = run_dir / "nodes"
                if not nodes_dir.exists():
                    continue
                for node_dir in nodes_dir.iterdir():
                    cont_path = node_dir / "continuation.json"
                    if cont_path.exists():
                        out.append(json.loads(cont_path.read_text(encoding="utf-8")))
            return out

        return await asyncio.to_thread(_scan)

    async def clear(self) -> None:
        def _clear():
            for sub in ("runs", "index"):
                p = self.root / sub
                if p.exists():
                    for root, dirs, files in os.walk(p, topdown=False):
                        for f in files:
                            try:
                                os.remove(Path(root) / f)
                            except Exception:
                                logger = logging.getLogger(
                                    "aethergraph.services.continuations.stores.fs_store"
                                )
                                logger.warning("Failed to remove file: %s", Path(root) / f)
                        for d in dirs:
                            try:
                                os.rmdir(Path(root) / d)
                            except Exception:
                                logger = logging.getLogger(
                                    "aethergraph.services.continuations.stores.fs_store"
                                )
                                logger.warning("Failed to remove dir: %s", Path(root) / d)

        await asyncio.to_thread(_clear)

    async def alias_for(self, token: str) -> str | None:
        return token[:24]
