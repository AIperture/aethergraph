from __future__ import annotations

from typing import Any

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.services.continuations.continuation import Correlator


class QueueChannelAdapter(ChannelAdapter):
    """
    Generic adapter that writes OutEvents into a per-channel outbox in kv_hot.

    This is meant to be paired with:
      - /ws/channel   (to stream events to browser/clients)
      - optional /channel/outbox polling endpoint

    Capabilities: full superset to avoid downgrades in ChannelBus._smart_fallback.
    """

    # Slack-level capability set; user code can still choose to ignore some fields.
    capabilities: set[str] = {"text", "buttons", "image", "file", "edit", "stream"}

    def __init__(self, container, *, scheme: str = "ext"):
        self.c = container
        self.scheme = scheme

    async def send(self, event: OutEvent) -> dict | None:
        """
        Serialize OutEvent to a JSON-friendly dict and append to the channel outbox.

        Consumers (WS client, HTTP polling, etc.) can render this however they like.
        """
        ch_key = event.channel  # expected to already look like "ext:chan/<id>" or similar
        outbox_key = f"outbox://{ch_key}"

        # Minimal normalization; keep as much info as possible for UI.
        payload: dict[str, Any] = {
            "type": event.type,
            "channel": event.channel,
            "text": event.text,
            "meta": event.meta,
            "rich": event.rich,
            "upsert_key": event.upsert_key,
            "file": event.file,
            "buttons": [],
        }

        # Buttons: flatten for clients (label/value/style/url)
        if event.buttons:
            btns = []
            for b in event.buttons:
                btns.append(
                    {
                        "label": getattr(b, "label", None),
                        "value": getattr(b, "value", None),
                        "style": getattr(b, "style", None),
                        "url": getattr(b, "url", None),
                    }
                )
            payload["buttons"] = btns

        # simple timestamp if you have a clock service; otherwise omit
        if hasattr(self.c, "clock"):
            payload["ts"] = self.c.clock.now_ts()

        await self.c.kv_hot.list_append(outbox_key, [payload])

        return {
            "correlator": Correlator(
                scheme=self.scheme,
                channel=ch_key,
                thread=(event.meta or {}).get("thread") or "",
                message=None,
            )
        }
