# src/aethergraph/plugins/channel/adapters/webui.py
from __future__ import annotations
from typing import Deque, Dict, Set, Optional, Any 
from dataclasses import asdict, is_dataclass
from collections import deque

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.services.continuations.continuation import Correlator

class WebSessionHub:
    def __init__(self, backlog_size: int = 100):
        self._conns: Dict[str, Set] = {}
        self._backlog: Dict[str, Deque[dict]] = {}
        self._backlog_size = backlog_size

    async def attach(self, session_id: str, sender):
        self._conns.setdefault(session_id, set()).add(sender)
        # flush backlog to this new connection
        for payload in list(self._backlog.get(session_id, [])):
            try:
                await sender(payload)
            except Exception:
                pass  # ignore single-send failures

    async def detach(self, session_id: str, sender):
        s = self._conns.get(session_id)
        if s and sender in s:
            s.remove(sender)
            if not s:
                self._conns.pop(session_id, None)

    async def emit(self, session_id: str, payload: dict):
        conns = list(self._conns.get(session_id, []))
        if conns:
            for send in conns:
                try:
                    await send(payload)
                except Exception:
                    await self.detach(session_id, send)
            return

        # no live connections → store to backlog
        q = self._backlog.setdefault(session_id, deque(maxlen=self._backlog_size))
        q.append(payload)

def _serialize_event(event: OutEvent) -> dict:
    """Dataclass → dict; normalize buttons; strip file bytes; drop None."""
    if is_dataclass(event):
        payload = asdict(event)
    else:
        # be lenient if a pydantic-like instance sneaks in
        payload = getattr(event, "model_dump", None) and event.model_dump() \
               or getattr(event, "dict", None) and event.dict() \
               or dict(event)

    # normalize buttons dict -> list
    btns = payload.get("buttons")
    if isinstance(btns, dict):
        payload["buttons"] = list(btns.values())

    # drop binary bytes from file
    f = payload.get("file")
    if isinstance(f, dict) and "bytes" in f:
        f = f.copy()
        f.pop("bytes", None)
        payload["file"] = f

    # clean None
    return {k: v for k, v in payload.items() if v is not None}

class WebChannelAdapter(ChannelAdapter):
    """
    Channel key: 'web:session/{session_id}'
    Mirrors Slack adapter semantics for:
      - correlators
      - stream/progress upserts via upsert_key
      - buttons/image/file payload shapes
    """
    capabilities: Set[str] = {"text","buttons","image","file","edit","stream"}

    def __init__(self, hub: WebSessionHub):
        self.hub = hub
        self._first_msg_by_key: Dict[tuple[str, str], str] = {}  # (channel, upsert_key) -> synthetic message id
        self._seq_by_chan: Dict[str, int] = {}

    @staticmethod
    def _parse(channel_key: str) -> dict:
        # "web:session/{id}" -> {"session": "..."}
        parts = channel_key.split(":", 1)[1]  # session/{id}
        k, v = parts.split("/", 1)
        return {k: v}

    def _next_seq(self, ch: str) -> str:
        n = self._seq_by_chan.get(ch, 0) + 1
        self._seq_by_chan[ch] = n
        return str(n)

    async def peek_thread(self, channel_key: str) -> Optional[str]:
        # no threads in web adapter
        return None

    async def send(self, event: OutEvent) -> dict | None:
        meta = self._parse(event.channel)
        session_id = meta["session"]

        # upsert bookkeeping like Slack: ensure a stable logical message per upsert_key
        if event.upsert_key:
            key = (event.channel, event.upsert_key)
            if key not in self._first_msg_by_key:
                self._first_msg_by_key[key] = self._next_seq(event.channel)

        payload: Dict[str, Any] = _serialize_event(event)
        await self.hub.emit(session_id, payload)

        # return correlator so ChannelSession can bind (consistent with Slack)
        return {
            "correlator": Correlator(
                scheme="web",
                channel=event.channel,
                thread=None,
                message=self._next_seq(event.channel),
            )
        }
