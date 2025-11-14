"""
Webhook channel adapter.

Channel key format (after alias resolution):
  webhook:<URL>

Use cases include:
- Sending notifications to generic webhook endpoints.
- Integrating with services like Zapier, IFTTT, Discord, etc.
"""

# aethergraph/channels/webhook.py
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any
import warnings

from aethergraph.contracts.services.channel import Button, ChannelAdapter, OutEvent
from aethergraph.plugins.net.http import get_async_client

logger = logging.getLogger("aethergraph.channels.webhook")


@dataclass
class WebhookChannelAdapter(ChannelAdapter):
    """
    Generic inform-only webhook adapter.

    Channel key:
      webhook:<URL>
    Examples:
      webhook:https://hooks.zapier.com/hooks/catch/123/abc/
      webhook:https://discord.com/api/webhooks/.../...
    """

    default_headers: dict[str, str] | None = None
    timeout_seconds: float = 10.0

    capabilities: set[str] = frozenset({"text", "file", "rich", "buttons"})

    def _url_for(self, channel_key: str) -> str:
        try:
            _, url = channel_key.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid webhook channel key: {channel_key!r}") from exc
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"Webhook channel key must contain a full URL, got: {url!r}")
        return url

    def _serialize_buttons(self, buttons: dict[str, Button] | None) -> list[dict[str, Any]]:
        if not buttons:
            return []
        return [
            {"key": k, "label": b.label, "value": b.value, "url": b.url, "style": b.style}
            for k, b in buttons.items()
        ]

    def _serialize_file(self, file_info: dict[str, Any] | None) -> dict[str, Any] | None:
        if not file_info:
            return None
        return {
            "name": file_info.get("filename") or file_info.get("name"),
            "mimetype": file_info.get("mimetype"),
            "url": file_info.get("url"),
            "size": file_info.get("size"),
        }

    def _build_payload(self, event: OutEvent) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
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
        # For Discord-like webhooks that expect `content`
        if event.text is not None:
            payload["content"] = event.text
        return payload

    async def send(self, event: OutEvent) -> None:
        url = self._url_for(event.channel)
        payload = self._build_payload(event)
        headers = {"Content-Type": "application/json", **(self.default_headers or {})}

        try:
            async with get_async_client(self.timeout_seconds, headers) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    body = resp.text
                    logger.debug(
                        f"[WebhookChannelAdapter] POST {url} -> HTTP {resp.status_code}. "
                        f"Body: {body[:300]!r}"
                    )
        except Exception as e:
            # Best-effort; don't bubble failures into graph control flow
            warnings.warn(f"[WebhookChannelAdapter] Failed to POST to {url}: {e}", stacklevel=2)
