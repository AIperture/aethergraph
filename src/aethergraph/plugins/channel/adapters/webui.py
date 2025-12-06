from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from aethergraph.contracts.services.channel import Button, ChannelAdapter, OutEvent
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.continuations.continuation import Correlator


@dataclass
class UIChannelEvent:
    id: str
    run_id: str
    channel_key: str
    type: str  # original OutEvent.type, e.g. "agent.message"
    text: str | None
    buttons: list[dict[str, Any]]
    file: dict[str, Any] | None
    meta: dict[str, Any]
    ts: float


class WebUIChannelAdapter(ChannelAdapter):
    """
    Channel adapter for the AetherGraph web UI.

    - channel_key format (d0): "ui:run/<run_id>"
    - Writes normalized UI events into EventLog with scope_id=f"run-ui:{run_id}"
    """

    capabilities: set[str] = {"text", "buttons", "file", "stream", "edit"}

    def __init__(self, event_log: EventLog):
        self.event_log = event_log

    def _extract_run_id(self, channel_key: str) -> str:
        """
        Parse "ui:run/<run_id>" channel key to get run_id.
        """
        try:
            scheme, rest = channel_key.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid UI channel key: {channel_key!r}") from exc

        if scheme != "ui":
            raise ValueError(f"Invalid UI channel key scheme: {scheme!r}")

        # for d0, expect "run/<run_id>"
        if not rest.startswith("run/"):
            # future-compat: allow direct run_id, but warn
            # e.g. "ui:<run_id>"
            return rest  # assume rest is run_id directly

        return rest.split("/", 1)[1]  # get run_id

    def _button_to_dict(self, b: Button | Any) -> dict[str, Any]:
        # Be defensive: Button is a dataclass, but Slack adapter also handles light-weight objects
        return {
            "label": getattr(b, "label", None),
            "value": getattr(b, "value", None),
            "style": getattr(b, "style", None),
            "url": getattr(b, "url", None),
        }

    async def send(self, event: OutEvent) -> dict | None:
        """
        Normalize OutEvent -> UIChannelEvent dict and append to EventLog.
        """
        run_id = self._extract_run_id(event.channel)

        raw_buttons = getattr(event, "buttons", None) or []
        buttons = [self._button_to_dict(b) for b in raw_buttons]

        file_info = getattr(event, "file", None) or None

        scope_id = event.meta.get("run_id") if event.meta else None
        if not scope_id and run_id:
            scope_id = run_id

        row = {
            "id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).timestamp(),
            "scope_id": scope_id,
            "kind": "run_channel",
            "payload": {
                "type": event.type,
                "text": event.text,
                "buttons": buttons,
                "file": file_info,
                "meta": event.meta or {},
            },
        }
        await self.event_log.append(row)

        # Optional correlator for consistency with other adapters
        return {
            "run_id": run_id,
            "correlator": Correlator(
                scheme="ui",
                channel=event.channel,
                thread="",
                message=None,
            ),
        }
