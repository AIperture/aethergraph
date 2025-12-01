from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aethergraph.services.channel.ingress import ChannelIngress, IncomingFile, IncomingMessage

router = APIRouter()


class RunChannelIncomingBody(BaseModel):
    """
    Inbound message from AG web UI to a run's channel.

    This is a thin wrapper over ChannelIncomingBody, but we **infer**:
      - scheme = "ui"
      - channel_id = f"run/{run_id}"
    """

    text: str | None = None
    # future: allow UI to attach artifacts or URLs we can convert
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
      {
        "text": "hello",
        "meta": {...}
      }
    Backend maps this to ChannelIngress with:
      scheme="ui", channel_id=f"run/{run_id}"
    """
    try:
        container = request.app.state.container  # type: ignore
        ingress: ChannelIngress = container.channel_ingress

        # normalize files, if any (for now we expect only URL/uri-style descriptors)
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

        ok = await ingress.handle(
            IncomingMessage(
                scheme="ui",
                channel_id=f"run/{run_id}",  # d0 convention
                thread_id=None,  # can be extended later
                text=body.text,
                files=files,
                choice=body.choice,
                meta=body.meta or {},
            )
        )
        return JSONResponse({"ok": True, "resumed": ok})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
