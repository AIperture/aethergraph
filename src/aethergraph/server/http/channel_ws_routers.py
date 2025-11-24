from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/channel")
async def ws_channel(ws: WebSocket):
    """
    Generic outbound event stream.

    Client must first send a JSON handshake:
      {"scheme": "ext", "channel_id": "user-123"}

    Then we stream any events that the queue-based ChannelAdapter
    appends to `outbox://<scheme>:chan/<channel_id>`.
    """
    await ws.accept()

    hello = await ws.receive_json()
    scheme = hello.get("scheme") or "ext"
    channel_id = hello["channel_id"]

    container = ws.app.state.container
    c = container

    ch_key = f"{scheme}:chan/{channel_id}"
    outbox_key = f"outbox://{ch_key}"

    last_idx = 0
    try:
        while True:
            await asyncio.sleep(0.25)
            events = await c.kv_hot.list_get(outbox_key) or []
            if last_idx < len(events):
                for ev in events[last_idx:]:
                    # ev is a dict produced by our queue-based adapter
                    await ws.send_json(ev)
                last_idx = len(events)
    except WebSocketDisconnect:
        # just drop; nothing special needed
        return
