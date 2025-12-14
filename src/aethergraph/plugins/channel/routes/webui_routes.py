from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.core.runtime.run_manager_local import RunManager
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.channel.ingress import ChannelIngress, IncomingFile, IncomingMessage
from aethergraph.services.registry.unified_registry import UnifiedRegistry

router = APIRouter()


class RunChannelIncomingBody(BaseModel):
    """
    Inbound message from AG web UI to a run's channel.
    """

    text: str | None = None
    files: list[dict[str, Any]] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None


class SessionChatIncomingBody(BaseModel):
    """
    Inbound message from AG web UI to a session's chat channel.
    """

    text: str | None = None
    files: list[dict[str, Any]] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None
    # optional
    agent_id: str | None = None


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
        event_log = container.eventlog

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


@router.post("/sessions/{session_id}/chat/incoming")
async def session_chat_incoming(
    session_id: str,
    body: SessionChatIncomingBody,
    request: Request,
    identity: RequestIdentity = Depends(get_identity),  # noqa B008
) -> JSONResponse:
    """
    Inbound chat message for a session.

    - Logs a `user.message` event under scope_id=session_id, kind="session_chat".
    - Attempts to resume any waiting continuation on the session channel.
    - If nothing resumed, spawns a new agent run for this session.
    """
    try:
        print("🍎 session_chat_incoming called with session_id=", session_id)
        container = current_services()
        print("🍎 obtained container:", container)
        ingress: ChannelIngress = container.channel_ingress
        event_log = container.eventlog
        rm: RunManager = container.run_manager
        registry: UnifiedRegistry = container.registry

        print("🍎 obtained services: ingress, event_log, run_manager, registry")
        # 1) Normalize files into IncomingFile list (future use)
        files: list[IncomingFile] = []
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
        print("🍎 normalized files:", files)
        text = body.text or body.choice or ""
        if text:
            now_ts = datetime.now(timezone.utc).timestamp()
            row = {
                "id": str(uuid4()),
                "ts": now_ts,
                "scope_id": session_id,
                "kind": "session_chat",
                "payload": {
                    "type": "user.message",
                    "text": text,
                    "buttons": [],
                    "file": None,
                    "meta": {
                        **(body.meta or {}),
                        "direction": "inbound",
                        "role": "user",
                    },
                },
            }
            await event_log.append(row)

        print("🍎 logged user.message event for session:", session_id)
        # 2) Try to resume any waiting continuation via ChannelIngress
        resumed = await ingress.handle(
            IncomingMessage(
                scheme="ui",
                channel_id=f"session/{session_id}",
                thread_id=None,
                text=body.text,
                files=files,
                choice=body.choice,
                meta=body.meta or {},
            )
        )

        # 3) If nothing resumed, spawn a new agent run for this session
        run_id: str
        if not resumed:
            agent_id = body.agent_id
            if agent_id is None:
                # for v1 it is fine to require frontend to specify agent_id
                # later we can derive default agent per session
                raise HTTPException(
                    status_code=400,
                    detail="agent_id is required when no continuation is resumed",
                )

            # Resolve agent meta -> backing graph
            agent_meta = registry.get_meta(nspace="agent", name=agent_id)
            print("🍎 resolved agent_meta:", agent_meta)
            if not agent_meta:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent not found: {agent_id}",
                )

            backing = agent_meta.get("backing", {})
            if backing.get("type") != "graphfn":
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported agent backing type: {backing.get('type')}. Only 'graphfn' is supported in v1.",
                )

            graph_id = backing["name"]
            # build inputs for the agent graph -- in agent case, we pass message + files
            inputs = {
                "message": text,
                "files": files,
                "session_id": session_id,  # for convenience, we can derive session inside graph too
                "user_meta": body.meta or {},  # optional user meta
                # later we can add "context_refs": [...] if we do context retrieval from session history
            }

            record = await rm.submit_run(
                graph_id=graph_id,
                inputs=inputs,
                session_id=session_id,
                identity=identity,
                origin=RunOrigin.chat,
                visibility=RunVisibility.normal,
                importance=RunImportance.normal,
                agent_id=agent_id,
                app_id=agent_meta.get("app_id"),  # optional, if you attach this
                tags=["session:" + session_id, "agent:" + agent_id],
            )
            run_id = record.run_id

        return JSONResponse(
            {"ok": True, "resumed": resumed, "run_id": run_id if not resumed else None}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
