"""
Webhook channel adapter.

Channel key format (after alias resolution):
  webhook:<URL>

Use cases include:
- Sending notifications to generic webhook endpoints.
- Integrating with services like Zapier, IFTTT, Discord, etc.
"""


import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Set
import warnings

import aiohttp

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent, Button
import logging


@dataclass
class WebhookChannelAdapter(ChannelAdapter):
    """
    Generic inform-only webhook adapter.

    Channel key format (after alias resolution):
      webhook:<URL>

    Examples:
      webhook:https://hooks.zapier.com/hooks/catch/123/abc/
      webhook:https://discord.com/api/webhooks/.../...
    """

    default_headers: Dict[str, str]
    timeout_seconds: float
    _session: aiohttp.ClientSession | None = None

    capabilities: Set[str] = frozenset({"text", "file", "rich", "buttons"})

    def __init__(
        self,
        default_headers: Dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
    ):
        self.default_headers = default_headers or {}
        self.timeout_seconds = timeout_seconds
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _url_for(self, channel_key: str) -> str:
        # channel_key = "webhook:https://example.com/hook"
        try:
            _, url = channel_key.split(":", 1)
        except ValueError:
            raise ValueError(f"Invalid webhook channel key: {channel_key!r}")
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"Webhook channel key must contain a full URL, got: {url!r}")
        return url

    def _serialize_buttons(self, buttons: Dict[str, Button] | None) -> list[Dict[str, Any]]:
        if not buttons:
            return []
        out: list[Dict[str, Any]] = []
        for key, b in buttons.items():
            out.append(
                {
                    "key": key,
                    "label": b.label,
                    "value": b.value,
                    "url": b.url,
                    "style": b.style,
                }
            )
        return out

    def _serialize_file(self, file_info: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not file_info:
            return None
        return {
            "name": file_info.get("filename") or file_info.get("name"),
            "mimetype": file_info.get("mimetype"),
            "url": file_info.get("url"),
            "size": file_info.get("size"),
        }

    def _build_payload(self, event: OutEvent) -> Dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "type": event.type,
            "channel": event.channel,
            "text": event.text,
            "meta": event.meta or {},
            "rich": event.rich or {},
            "buttons": self._serialize_buttons(event.buttons),
            "file": self._serialize_file(event.file),
            "upsert_key": event.upsert_key,
            "timestamp": ts,
        }
       # for Discord / other webhook UIs that expect `content`
        if event.text is not None:
            payload["content"] = event.text
        return payload


    async def send(self, event: OutEvent) -> None:
        url = self._url_for(event.channel)
        payload = self._build_payload(event)
        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            **self.default_headers,
        }

        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger = logging.getLogger("aethergraph.plugins.channel.adapters.webhook")
                    logger.debug(f"[WebhookChannelAdapter] POST {url} -> HTTP {resp.status}. Body: {body[:300]!r}")

        except Exception as e:
            # Best-effort; we don't propagate errors back to the graph
            warnings.warn(f"[WebhookChannelAdapter] Failed to POST to {url}: {e}")
