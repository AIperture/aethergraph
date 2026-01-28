from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from aethergraph.contracts.services.channel import Button, ChannelAdapter, OutEvent
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.channel.event_hub import EventHub
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
    files: list[dict[str, Any]] | None = None  # optional
    rich: dict[str, Any] | None = None  # optional
    upsert_key: str | None = None  # optional


class WebUIChannelAdapter(ChannelAdapter):
    """
    Channel adapter for the AetherGraph web UI.

    - channel_key format (d0): "ui:run/<run_id>"
    - Writes normalized UI events into EventLog with scope_id=f"run-ui:{run_id}"
    """

    capabilities: set[str] = {"text", "buttons", "file", "stream", "edit"}

    def __init__(self, event_log: EventLog, event_hub: EventHub | None = None) -> None:
        self.event_log = event_log
        self.event_hub = event_hub

    def _extract_target(self, channel_key: str) -> tuple[str, str]:
        """
        Parse "ui:run/<run_id>" or "ui:session/<session_id>".

        Returns (scope_kind, id), where scope_kind in {"run", "session"}.
        """
        try:
            scheme, rest = channel_key.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid UI channel key: {channel_key!r}") from exc

        if scheme != "ui":
            raise ValueError(f"Invalid UI channel key scheme: {scheme!r}")

        if rest.startswith("run/"):
            return "run", rest.split("/", 1)[1]
        if rest.startswith("session/"):
            return "session", rest.split("/", 1)[1]

        # fallback: treat as run id
        return "run", rest

    def _button_to_dict(self, b: Button | Any) -> dict[str, Any]:
        # Be defensive: Button is a dataclass, but Slack adapter also handles light-weight objects
        return {
            "label": getattr(b, "label", None),
            "value": getattr(b, "value", None),
            "style": getattr(b, "style", None),
            "url": getattr(b, "url", None),
        }

    async def send(self, event: OutEvent) -> dict | None:
        scope_kind, target_id = self._extract_target(event.channel)

        raw_buttons = getattr(event, "buttons", None) or []
        buttons = [self._button_to_dict(b) for b in raw_buttons]
        file_info = getattr(event, "file", None) or None

        files = getattr(event, "files", None) or None
        rich = getattr(event, "rich", None) or None
        upsert_key = getattr(event, "upsert_key", None)

        meta = event.meta or {}
        agent_id = meta.get("agent_id") or meta.get("agent")
        if agent_id:
            meta["agent_id"] = agent_id

        session_id = meta.get("session_id")
        run_id = meta.get("run_id")

        if scope_kind == "session":
            scope_id = session_id or target_id
            kind = "session_chat"
        else:
            scope_id = run_id or target_id
            kind = "run_channel"

        # ------------------------------------------------------------
        # ✅ STREAMING POLICY:
        # - Do NOT persist start/delta
        # - Persist end as a final agent.message
        # ------------------------------------------------------------
        ephemeral_stream_types = {"agent.stream.start", "agent.stream.delta"}
        is_ephemeral_stream = event.type in ephemeral_stream_types
        is_stream_end = event.type == "agent.stream.end"

        payload_type = event.type
        payload_text = event.text

        # Persist stream.end as an agent.message (final)
        if is_stream_end:
            payload_type = "agent.message"
            # Mark it so UI/debug can know it came from a stream
            meta = {**meta, "_stream_final": True, "_stream_type": "end"}

        row = {
            "id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).timestamp(),
            "scope_id": scope_id,
            "kind": kind,
            "payload": {
                "type": payload_type,
                "text": payload_text,
                "buttons": buttons,
                "file": file_info,
                "files": files,
                "rich": rich,
                "upsert_key": upsert_key,
                "meta": meta,
                "agent_id": meta.get("agent_id"),
            },
        }

        # ✅ Only persist non-ephemeral stream events
        if not is_ephemeral_stream:
            await self.event_log.append(row)

        # ✅ Always broadcast if hub exists (so streaming works)
        if self.event_hub is not None:
            await self.event_hub.broadcast(row)

        return {
            "run_id": run_id or target_id,
            "correlator": Correlator(scheme="ui", channel=event.channel, thread="", message=None),
        }
