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
        api_key: str | None = None,  # currently unused
        http_client: httpx.AsyncClient | None = None,  # managed externally if provided
        ws_path: str = "/ws/channel",  # currently unused
    ):
        self.base_url = base_url
        self.scheme = scheme
        self.channel_id = channel_id
        self.thread_id = thread_id
        self.timeout = timeout
        self.api_key = api_key
        self.ws_path = ws_path

        self._external_client = http_client
        self._client: httpx.AsyncClient | None = None
        self._owns_client = http_client is None

    # ------------- internal helpers -------------
    @property
    def client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def aclose(self):
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _default_thread_id(self, thread_id: str | None) -> str | None:
        return thread_id if thread_id is not None else self.thread_id

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an httpx.AsyncClient."""
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

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
        r = await self.client.post(url, json=payload)
        r.raise_for_status()
        return r.json

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
        r = await self.client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

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
        r = await self.client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

    async def resume(
        self,
        run_id: str,
        node_id: str,
        token: str,
        resume_key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Low-level manual resume via /channel/resume.
        """
        url = f"{self.base_url}/channel/resume"
        body = {
            "run_id": run_id,
            "node_id": node_id,
            "token": token,
            "resume_key": resume_key,
            "payload": payload or {},
        }
        r = await self.client.post(url, json=body)
        r.raise_for_status()
        return r.json()

    # --------- Outbound from AG (WebSocket) ---------
    async def iter_events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Receive outbound channel events over a WebSocket.

        Expected server endpoint: /ws/channel

        Query params:
          - scheme
          - channel_id
          - thread_id (optional)
          - api_key (optional)
        """
        # Build ws URL from base_url (http/https -> ws/wss)
        if self.base_url.startswith("https://"):
            ws_base = "wss://" + self.base_url[len("https://") :]
        elif self.base_url.startswith("http://"):
            ws_base = "ws://" + self.base_url[len("http://") :]
        else:
            # assume ws already
            ws_base = self.base_url

        params = {
            "scheme": self.scheme,
            "channel_id": self.channel_id,
        }
        if self.thread_id:
            params["thread_id"] = self.thread_id
        if self.api_key:
            params["api_key"] = self.api_key

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{ws_base}{self.ws_path}?{query}"

        async with websockets.connect(url) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    data = {"raw": msg}
                yield data
