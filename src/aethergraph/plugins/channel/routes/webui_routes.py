# src/aethergraph/server/webui.py
from __future__ import annotations
import os
from typing import Optional, Callable, Awaitable, Any, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from aethergraph.plugins.channel.adapters.webui import WebSessionHub, WebChannelAdapter
from aethergraph.services.continuations.continuation import Correlator

webui_router = APIRouter()

# ------- runtime singletons (attached in create_app) -------
HUB_ATTR = "web_session_hub"
UPLOAD_DIR_ATTR = "web_upload_dir"

def _hub(app) -> WebSessionHub:
    return getattr(app.state, HUB_ATTR)

def _uploads_dir(app) -> str:
    return getattr(app.state, UPLOAD_DIR_ATTR)

# ------- WebSocket endpoint -------
@webui_router.websocket("/ws/channel/{session_id}")
async def ws_channel(ws: WebSocket, session_id: str):
    await ws.accept()

    async def send_json(payload: dict):
        await ws.send_json(payload)

    hub = _hub(ws.app)
    await hub.attach(session_id, send_json)

    try:
        while True:
            msg = await ws.receive_json()
            # Expect inbound: {"type": "resume", "run_id": ..., "node_id": ..., "token": ..., "payload": {...}}
            t = (msg or {}).get("type")
            if t == "resume":
                c = ws.app.state.container
                # basic token verification happens in ResumeRouter
                await c.resume_router.resume(
                    run_id=msg["run_id"],
                    node_id=msg["node_id"],
                    token=msg["token"],
                    payload=msg.get("payload") or {},
                )
            # optionally handle ping or upload notifications (not required)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.detach(session_id, send_json)

# ------- HTTP resume fallback (for InputDock before WS ready) -------
class ResumeBody(BaseModel):
    run_id: str
    node_id: str
    token: str
    payload: dict

@webui_router.post("/api/web/resume")
async def http_resume(request: Request, body: ResumeBody):
    c = request.app.state.container
    await c.resume_router.resume(body.run_id, body.node_id, body.token, body.payload)
    return {"ok": True}

# ------- Uploads -------
@webui_router.post("/api/web/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    """
    Save to <workspace>/web_uploads/<session_or_any>/... and return FileRef[]:
      [{url, filename, size, mime}]
    UI doesn't pass session; we just save under a common folder.
    """
    root = _uploads_dir(request.app)
    os.makedirs(root, exist_ok=True)

    out = []
    for f in files:
        target = os.path.join(root, f.filename)
        with open(target, "wb") as w:
            w.write(await f.read())
        url = f"/api/web/files/{f.filename}"
        out.append({"url": url, "filename": f.filename, "mime": f.content_type or "application/octet-stream"})
    return out

@webui_router.get("/api/web/files/{filename}")
async def serve_uploaded(request: Request, filename: str):
    root = _uploads_dir(request.app)
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, filename=filename)

# ------- Integration helper -------
def install_web_channel(app: Any):
    """
    1) Creates a WebSessionHub
    2) Registers WebChannelAdapter under prefix 'web' in ChannelBus
    3) Sets default channel to 'web:session/<uuid>'
    4) Ensures upload dir exists
    """
    # 1) Hub
    hub = WebSessionHub()
    setattr(app.state, HUB_ATTR, hub)

    # 2) Adapter registration
    container = app.state.container
    web_adapter = WebChannelAdapter(hub)
    container.channels.adapters["web"] = web_adapter

    # 3) Keep default as console unless you want to swap globally:
    # container.channels.set_default_channel_key("web:session/dev-local")

    # 4) Upload dir
    updir = os.path.join(container.root, "web_uploads")
    os.makedirs(updir, exist_ok=True)
    setattr(app.state, UPLOAD_DIR_ATTR, updir)
