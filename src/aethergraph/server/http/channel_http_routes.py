from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aethergraph.services.channel.ingress import ChannelIngress, IncomingFile, IncomingMessage

router = APIRouter()


# --------- Pydantic models for HTTP request/response ---------
class HttpIncomingFile(BaseModel):
    id: str | None = None
    name: str | None = None
    mimetype: str | None = None
    size: int | None = None
    url: str | None = None
    uri: str | None = None
    extra: dict[str, Any] | None = None


class ChannelIncomingBody(BaseModel):
    """
    High-level resume via channel (no run_id/node_id/token exposed).
    """

    scheme: str = "ext"
    channel_id: str
    thread_id: str | None = None

    text: str | None = None
    files: list[HttpIncomingFile] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None


class ChannelManualResumeBody(BaseModel):
    """
    Low-level resume for power users: explicit run/node/token.
    """

    run_id: str
    node_id: str
    token: str
    payload: dict[str, Any] | None = None


# --------- HTTP route handlers ---------
@router.post("/channel/incoming")
async def channel_incoming(body: ChannelIncomingBody, request: Request):
    """
    Generic inbound message endpoint. Typical UI call looks like:

      POST /channel/incoming
      {
        "scheme": "ext",
        "channel_id": "user-123",
        "text": "hello",
        "meta": {"foo": "bar"}
      }
    """
    try:
        container = request.app.state.container
        ingress: ChannelIngress = container.channel_ingress  # TODO: wire via default container

        files = []
        if body.files:
            files = [
                IncomingFile(
                    id=f.id,
                    name=f.name,
                    mimetype=f.mimetype,
                    size=f.size,
                    url=f.url,
                    uri=f.uri,
                    extra=f.extra or {},
                )
                for f in body.files
            ]

        ok = await ingress.handle(
            IncomingMessage(
                scheme=body.scheme,
                channel_id=body.channel_id,
                thread_id=body.thread_id,
                text=body.text,
                files=files,
                choice=body.choice,
                meta=body.meta or {},
            )
        )
        return JSONResponse({"ok": True, "resumed": ok})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/channel/resume")
async def channel_resume(body: ChannelManualResumeBody, request: Request):
    """
    Low-level resume for power users: explicit run/node/token.
    """
    try:
        container = request.app.state.container

        await container.resume_router.resume(
            run_id=body.run_id,
            node_id=body.node_id,
            token=body.token,
            payload=body.payload or {},
        )
        return JSONResponse({"ok": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
