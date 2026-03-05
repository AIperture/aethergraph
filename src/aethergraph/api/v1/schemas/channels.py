from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field  # type: ignore


class ChannelIngressRequest(BaseModel):
    kind: str = "chat_user"
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChannelEvent(BaseModel):
    event_id: str
    channel_id: str
    kind: str
    created_at: datetime
    data: dict[str, Any]


class ChannelEventListResponse(BaseModel):
    events: list[ChannelEvent]
    next_cursor: str | None = None
