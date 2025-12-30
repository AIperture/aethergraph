from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aethergraph.api.v1.deps import RequestIdentity  # adjust import
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.eventhub.event_hub import EventHub

"""
WebSocket endpoint for pushing EventLog rows to the browser in real time.

Protocol (JSON messages from client):

  { "type": "subscribe",
    "scope_id": "session:<id>",
    "kinds": ["session_chat"] }

  { "type": "unsubscribe",
    "scope_id": "session:<id>",
    "kinds": ["session_chat"] }

  { "type": "ping" }

Messages from server:

  { "type": "event",
    "scope_id": "session:<id>",
    "kind": "session_chat",
    "id": "<event-id>",
    "ts": <float>,
    "payload": { ...same as HTTP /chat/events... }
  }

  { "type": "pong" }

NOTE: This is a scaffold. It is *not* yet wired into auth or your router.
"""


router = APIRouter()


@router.websocket("/ws/events")
async def ws_events(
    websocket: WebSocket,
    identity: RequestIdentity = None,  # TODO: hook in proper auth if desired
) -> None:
    """
    WebSocket endpoint for UI event streaming.

    Typical usage (client-side, future):

      ws = new WebSocket("wss://.../ws/events");
      ws.send(JSON.stringify({
        type: "subscribe",
        scope_id: "session:<session_id>",
        kinds: ["session_chat"],
      }));

    For now this is scaffold-only and not used by the frontend.
    """
    await websocket.accept()

    container = current_services()
    event_hub: EventHub | None = getattr(container, "event_hub", None)

    if event_hub is None:
        # If EventHub hasn't been wired yet, just close gracefully.
        await websocket.close(code=1011)
        return

    # (scope_id, kind) -> callback
    callbacks: dict[tuple[str, str], Any] = {}

    # Queue of rows to send to this client
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def make_callback(scope_id: str, kind: str):
        async def _cb(row: dict) -> None:
            """
            Called by EventHub.broadcast(row).

            We avoid calling websocket.send_json directly here to keep ordering
            and error-handling in a single place (the sender task).
            """
            await queue.put(row)

        return _cb

    async def sender() -> None:
        """
        Background task that forwards rows from the queue to the WebSocket.
        """
        try:
            while True:
                row = await queue.get()
                # Minimal envelope; payload matches HTTP /chat/events structure.
                await websocket.send_json(
                    {
                        "type": "event",
                        "scope_id": row.get("scope_id"),
                        "kind": row.get("kind"),
                        "id": row.get("id"),
                        "ts": row.get("ts"),
                        "payload": row.get("payload") or {},
                    }
                )
        except WebSocketDisconnect:
            # Client went away; main function will handle cleanup.
            return
        except Exception:
            # TODO: log error
            return

    sender_task = asyncio.create_task(sender())

    async def subscribe(scope_id: str, kinds: list[str]) -> None:
        for kind in kinds:
            key = (scope_id, kind)
            if key in callbacks:
                continue
            cb = await make_callback(scope_id, kind)
            callbacks[key] = cb
            event_hub.subscribe(scope_id, kind, cb)

    async def unsubscribe(scope_id: str, kinds: list[str]) -> None:
        for kind in kinds:
            key = (scope_id, kind)
            cb = callbacks.pop(key, None)
            if cb is not None:
                event_hub.unsubscribe(scope_id, kind, cb)

    try:
        while True:
            msg = await websocket.receive_json()

            msg_type = msg.get("type")
            if msg_type == "subscribe":
                scope_id = msg["scope_id"]
                kinds = msg.get("kinds") or ["session_chat"]
                # TODO: enforce authorization here based on `identity` & scope_id
                await subscribe(scope_id, kinds)

            elif msg_type == "unsubscribe":
                scope_id = msg["scope_id"]
                kinds = msg.get("kinds") or ["session_chat"]
                await unsubscribe(scope_id, kinds)

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            # else: ignore unknown types for now

    except WebSocketDisconnect:
        # Normal disconnect
        pass
    except Exception:
        # TODO: log error
        pass
    finally:
        # Cleanup subscriptions and sender task
        for (scope_id, kind), cb in callbacks.items():
            event_hub.unsubscribe(scope_id, kind, cb)
        callbacks.clear()

        sender_task.cancel()
        with contextlib.suppress(Exception):
            await sender_task
