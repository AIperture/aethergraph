from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.channel.ingress import ChannelIngress, IncomingFile, IncomingMessage

router = APIRouter()


class RunChannelIncomingBody(BaseModel):
    """
    Inbound message from AG web UI to a run's channel.
    """

    text: str | None = None
    files: list[dict[str, Any]] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None


@router.post("/runs/{run_id}/channel/incoming")
async def run_channel_incoming(
    run_id: str,
    body: RunChannelIncomingBody,
    request: Request,
) -> JSONResponse:
    """
    Specialized ingress for AG Web UI.

    UI calls:
      POST /runs/<run_id>/channel/incoming
      { "text": "hello", "meta": {...} }

    Backend maps this to ChannelIngress with:
      scheme="ui", channel_id=f"run/{run_id}"
    and logs a `user.message` event into EventLog so the UI can render it.
    """
    try:
        container = request.app.state.container  # type: ignore
        ingress: ChannelIngress = container.channel_ingress
        event_log: EventLog = container.eventlog

        # 1) Normalize files into IncomingFile list (future use)
        files = []
        if body.files:
            for f in body.files:
                files.append(
                    IncomingFile(
                        id=f.get("id"),
                        name=f.get("name"),
                        mimetype=f.get("mimetype"),
                        size=f.get("size"),
                        url=f.get("url"),
                        uri=f.get("uri"),
                        extra=f.get("extra") or {},
                    )
                )

        # 2) Log the inbound user message **first**
        text = body.text or body.choice or ""
        if text:
            now_ts = datetime.now(timezone.utc).timestamp()
            row = {
                "id": str(uuid4()),
                "ts": now_ts,
                "scope_id": run_id,
                "kind": "run_channel",
                "payload": {
                    "type": "user.message",
                    "text": text,
                    "buttons": [],
                    "file": None,
                    "meta": {
                        **(body.meta or {}),
                        "direction": "inbound",
                        "role": "user",
                        # we don't yet know "resumed" here; can add later if needed
                    },
                },
            }
            await event_log.append(row)

        # 3) Now resume any waiting continuation via ChannelIngress
        resumed = await ingress.handle(
            IncomingMessage(
                scheme="ui",
                channel_id=f"run/{run_id}",
                thread_id=None,
                text=body.text,
                files=files,
                choice=body.choice,
                meta=body.meta or {},
            )
        )

        return JSONResponse({"ok": True, "resumed": resumed})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
