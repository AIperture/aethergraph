from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
import json
from typing import Any

import httpx
import websockets


class ChannelClient:
    """
    Convenience client for talking to a running AetherGraph server from Python.

    - send_* methods: external -> AG (inbound to AG via /channel/incoming)
    - iter_events():  AG -> external (outbound from AG via /ws/channel)

    This is intentionally thin; real apps can wrap it with their own abstractions.
    """

    def __init__(
        self,
        base_url: str,
        *,
        scheme: str = "ext",
        channel_id: str = "default",
        thread_id: str | None = None,
        timeout: float = 100.0,
    ):
        self.base_url = base_url
        self.scheme = scheme
        self.channel_id = channel_id
        self.thread_id = thread_id
        self.timeout = timeout

    # --------- Inbound to AG (HTTP) ---------
    async def send_text(self, text: str, *, meta: dict[str, Any] | None = None) -> httpx.Response:
        """
        Send a text message into AG via /channel/incoming.
        """
        url = f"{self.base_url}/channel/incoming"
        payload = {
            "scheme": self.scheme,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "text": text,
            "meta": meta or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp

    async def send_choice(
        self, choice: str, *, meta: dict[str, Any] | None = None
    ) -> httpx.Response:
        """
        Send a choice/approval response into AG via /channel/incoming.
        """
        url = f"{self.base_url}/channel/incoming"
        payload = {
            "scheme": self.scheme,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "choice": choice,
            "meta": meta or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp

    async def send_text_and_files(
        self,
        text: str | None,
        files: Iterable[dict[str, Any]],
        *,
        meta: dict[str, Any] | None = None,
    ):
        """
        Send a text message with attached files into AG via /channel/incoming.

        Each file is a dict with keys like:
          - name (str): filename
          - mimetype (str): MIME type
          - size (int): size in bytes
          - url (str): public URL to download the file
        """
        url = f"{self.base_url}/channel/incoming"
        payload = {
            "scheme": self.scheme,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "text": text,
            "files": list(files),
            "meta": meta or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp

    async def resume_manual(
        self, run_id: str, node_id: str, token: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """
        Low-level manual resume via /channel/manual_resume.
        """
        url = f"{self.base_url}/channel/manual_resume"
        body = {
            "run_id": run_id,
            "node_id": node_id,
            "token": token,
            "payload": payload or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=body)
        resp.raise_for_status()
        return resp

    # --------- Outbound from AG (WebSocket) ---------
    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator of events from /ws/channel.

        Each event is the JSON dict produced by QueueChannelAdapter.send:
          {
            "type": "...",
            "text": "...",
            "meta": {...},
            "rich": {...},
            "buttons": [...],
            "file": {...},
            "upsert_key": "...",
            "ts": ...
          }
        """
        # naive "http" -> "ws" replacement; tweak if you have https/wss
        if self.base_url.startswith("https://"):
            ws_base = "wss://" + self.base_url[len("https://") :]
        elif self.base_url.startswith("http://"):
            ws_base = "ws://" + self.base_url[len("http://") :]
        else:
            ws_base = self.base_url  # assume already ws/wss

        ws_url = ws_base + "/ws/channel"

        async with websockets.connect(ws_url) as ws:
            # handshake
            await ws.send(
                json.dumps(
                    {
                        "scheme": self.scheme,
                        "channel_id": self.channel_id,
                    }
                )
            )
            async for raw in ws:
                yield json.loads(raw)
