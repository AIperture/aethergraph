# stub, to move the server.channels module here later


from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from .deps import RequestIdentity, get_identity
from .schemas import (
    ChannelEvent,
    ChannelEventListResponse,
    ChannelIngressRequest,
)

router = APIRouter(tags=["channels"])


@router.post("/channels/{channel_id}/ingress")
async def channel_ingress(
    channel_id: str,
    req: ChannelIngressRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    """
    Ingest a message into a channel (HTTP).

    TODO:
      - Forward to Channel service / Correlator.
      - Likely emit a memory event + trigger continuations.
    """
    # Stub: just echo
    return {
        "channel_id": channel_id,
        "kind": req.kind,
        "text": req.text,
        "metadata": req.metadata,
        "user_id": identity.user_id,
    }


@router.get("/channels/{channel_id}/events", response_model=ChannelEventListResponse)
async def list_channel_events(
    channel_id: str,
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ChannelEventListResponse:
    """
    Polling-based channel event retrieval.

    TODO:
      - Integrate with channel/event store using cursor pagination.
    """
    now = datetime.utcnow()
    dummy = ChannelEvent(
        event_id="ch-evt-1",
        channel_id=channel_id,
        kind="chat_assistant",
        created_at=now - timedelta(seconds=30),
        data={"text": "Stub channel event"},
    )
    return ChannelEventListResponse(events=[dummy], next_cursor=None)


# ----- WebSocket for real-time -----


@router.websocket("/ws/channels/{channel_id}")
async def channel_websocket(
    websocket: WebSocket,
    channel_id: str,
):
    """
    WebSocket endpoint for real-time channel events.

    TODO:
      - Authenticate if needed (e.g., via query param token or headers).
      - Subscribe this socket to channel event stream.
      - Push events as they arrive; accept client messages as ingress.
    """
    await websocket.accept()
    try:
        # Very basic echo loop as a stub
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"[stub] Channel {channel_id} received: {data}")
    except WebSocketDisconnect:
        # TODO: clean up subscriptions if add them.
        pass
