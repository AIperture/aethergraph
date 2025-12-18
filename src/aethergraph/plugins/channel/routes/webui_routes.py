from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import json
import shutil
from typing import Any
import uuid
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.responses import JSONResponse

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.core.runtime.runtime_services import current_services
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


async def _save_upload_as_artifact(container, file: UploadFile, session_id: str) -> str:
    """
    Reads upload bytes, saves to ArtifactStore, returns the URI.
    Mirrors your Slack _stage_and_save logic.
    """
    # 1. Read bytes from the upload stream
    content = await file.read()

    # 2. Plan a staging path (using original extension)
    #    We use the session_id as the 'scope' or 'run_id' for organization
    file_ext = ""
    if "." in (file.filename or ""):
        file_ext = f".{file.filename.split('.')[-1]}"

    tmp_path = await container.artifacts.plan_staging_path(
        planned_ext=f"_{uuid.uuid4().hex[:8]}{file_ext}"
    )

    # 3. Write to temp local storage
    with open(tmp_path, "wb") as f:
        f.write(content)

    # 4. Save to Artifact Store
    #    We tag it with the session_id so we can find it later
    art = await container.artifacts.save_file(
        path=tmp_path,
        kind="upload",
        run_id=f"session_{session_id}",  # Or just session_id, depending on your convention
        graph_id="session_chat",  # Scope identifier
        node_id="user_input",
        tool_name="web.upload",
        tool_version="1.0.0",
        labels={"source": "web_chat", "original_name": file.filename, "session_id": session_id},
    )

    # Return the URI (e.g., "s3://...", "file://...", or "artifact://...")
    return getattr(art, "uri", None) or getattr(art, "path", None)


# @router.post("/sessions/{session_id}/chat/incoming")
# async def session_chat_incoming(
#     session_id: str,
#     body: SessionChatIncomingBody,
#     request: Request,
#     identity: RequestIdentity = Depends(get_identity),  # noqa B008
# ) -> JSONResponse:
#     """
#     Inbound chat message for a session.

#     - Logs a `user.message` event under scope_id=session_id, kind="session_chat".
#     - Attempts to resume any waiting continuation on the session channel.
#     - If nothing resumed, spawns a new agent run for this session.
#     """
#     try:
#         container = current_services()
#         ingress: ChannelIngress = container.channel_ingress
#         event_log = container.eventlog
#         rm: RunManager = container.run_manager
#         registry: UnifiedRegistry = container.registry

#         # 1) Normalize files into IncomingFile list (future use)
#         files: list[IncomingFile] = []
#         if body.files:
#             for f in body.files:
#                 files.append(
#                     IncomingFile(
#                         id=f.get("id"),
#                         name=f.get("name"),
#                         mimetype=f.get("mimetype"),
#                         size=f.get("size"),
#                         url=f.get("url"),
#                         uri=f.get("uri"),
#                         extra=f.get("extra") or {},
#                     )
#                 )
#         text = body.text or body.choice or ""
#         if text:
#             now_ts = datetime.now(timezone.utc).timestamp()
#             row = {
#                 "id": str(uuid4()),
#                 "ts": now_ts,
#                 "scope_id": session_id,
#                 "kind": "session_chat",
#                 "payload": {
#                     "type": "user.message",
#                     "text": text,
#                     "buttons": [],
#                     "file": None,
#                     "meta": {
#                         **(body.meta or {}),
#                         "direction": "inbound",
#                         "role": "user",
#                     },
#                 },
#             }
#             await event_log.append(row)

#         # 2) Try to resume any waiting continuation via ChannelIngress
#         resumed = await ingress.handle(
#             IncomingMessage(
#                 scheme="ui",
#                 channel_id=f"session/{session_id}",
#                 thread_id=None,
#                 text=body.text,
#                 files=files,
#                 choice=body.choice,
#                 meta=body.meta or {},
#             )
#         )

#         # 3) If nothing resumed, spawn a new agent run for this session
#         run_id: str
#         if not resumed:
#             agent_id = body.agent_id
#             if agent_id is None:
#                 # for v1 it is fine to require frontend to specify agent_id
#                 # later we can derive default agent per session
#                 raise HTTPException(
#                     status_code=400,
#                     detail="agent_id is required when no continuation is resumed",
#                 )

#             # Resolve agent meta -> backing graph
#             agent_meta = registry.get_meta(nspace="agent", name=agent_id)
#             if not agent_meta:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Agent not found: {agent_id}",
#                 )

#             run_vis_str = agent_meta.get(
#                 "run_visibility", RunVisibility.inline.value
#             )  # default inline
#             run_imp_str = agent_meta.get(
#                 "run_importance", RunImportance.ephemeral.value
#             )  # default ephemeral
#             run_vis = RunVisibility(run_vis_str)
#             run_imp = RunImportance(run_imp_str)

#             backing = agent_meta.get("backing", {})
#             if backing.get("type") != "graphfn":
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Unsupported agent backing type: {backing.get('type')}. Only 'graphfn' is supported in v1.",
#                 )

#             graph_id = backing["name"]
#             # build inputs for the agent graph -- in agent case, we pass message + files
#             inputs = {
#                 "message": text,
#                 "files": files,
#                 "session_id": session_id,  # for convenience, we can derive session inside graph too
#                 "user_meta": body.meta or {},  # optional user meta
#                 # later we can add "context_refs": [...] if we do context retrieval from session history
#             }

#             record = await rm.submit_run(
#                 graph_id=graph_id,
#                 inputs=inputs,
#                 session_id=session_id,
#                 identity=identity,
#                 origin=RunOrigin.chat,
#                 visibility=run_vis,
#                 importance=run_imp,
#                 agent_id=agent_id,
#                 app_id=agent_meta.get("app_id"),  # optional, if you attach this
#                 tags=["session:" + session_id, "agent:" + agent_id],
#             )
#             run_id = record.run_id

#         return JSONResponse(
#             {"ok": True, "resumed": resumed, "run_id": run_id if not resumed else None}
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e)) from e


async def _save_upload_as_artifact(container, upload: UploadFile, session_id: str) -> str:
    """
    Streams upload to disk, saves as artifact, returns URI.
    """
    filename = upload.filename or "unknown"
    ext = ""
    if "." in filename:
        ext = f".{filename.split('.')[-1]}"

    # 1. Plan Staging
    tmp_path = await container.artifacts.plan_staging_path(
        planned_ext=f"_{uuid.uuid4().hex[:6]}{ext}"
    )

    # 2. Save Bytes
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    # 3. Register Artifact
    artifact = await container.artifacts.save_file(
        path=tmp_path,
        kind="upload",
        run_id=f"session:{session_id}",
        graph_id="chat",
        node_id="user_input",
        tool_name="web.upload",
        tool_version="1.0.0",
        labels={
            "source": "web_chat",
            "original_name": filename,
            "session_id": session_id,
            "content_type": upload.content_type,
        },
    )

    # Return URI
    return getattr(artifact, "uri", None) or getattr(artifact, "path", None)


@router.post("/sessions/{session_id}/chat/incoming")
async def session_chat_incoming(
    session_id: str,
    # Form fields
    text: str = Form(""),
    agent_id: str | None = Form(None),  # noqa B008
    meta_json: str | None = Form(None),  # noqa B008
    # Files
    files: list[UploadFile] = File(default=[]),  # noqa B008
    # Context
    request: Request = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa B008
):
    container = current_services()
    ingress = container.channel_ingress
    registry = container.registry
    rm = container.run_manager
    event_log = container.eventlog

    # 1. Parse Meta
    meta: dict[str, Any] = {}
    if meta_json:
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid meta JSON") from e

    # 2. Process Files -> IncomingFile
    incoming_files: list[IncomingFile] = []

    for upload in files:
        # Save to artifact store
        uri = await _save_upload_as_artifact(container, upload, session_id)

        # Strict instantiation based on your dataclass
        incoming_files.append(
            IncomingFile(
                id=str(uuid.uuid4()),
                name=upload.filename,
                mimetype=upload.content_type,
                size=upload.size,
                url=None,  # Not using public URL
                uri=uri,  # The internal artifact URI
                extra={"source": "web_upload"},  # Matches dict[str, Any]
            )
        )

    # 3. Log Event
    # FIX: Use dataclasses.asdict() instead of .model_dump()
    if text or incoming_files:
        now_ts = datetime.now(timezone.utc).timestamp()

        # Convert dataclasses to dicts for JSON logging
        files_payload = [dataclasses.asdict(f) for f in incoming_files]

        await event_log.append(
            {
                "id": str(uuid.uuid4()),
                "ts": now_ts,
                "scope_id": session_id,
                "kind": "session_chat",
                "payload": {
                    "type": "user.message",
                    "text": text,
                    "files": files_payload,
                    "meta": {**meta, "direction": "inbound", "role": "user"},
                },
            }
        )

    # 4. Ingress Handle
    # incoming_files (List) satisfies Iterable[IncomingFile]
    resumed = await ingress.handle(
        IncomingMessage(
            scheme="ui",
            channel_id=f"session/{session_id}",
            thread_id=None,
            text=text,
            files=incoming_files if incoming_files else None,
            meta=meta,
        )
    )

    # 5. Spawn Run (if needed)
    run_id = None
    if not resumed:
        if not agent_id:
            raise HTTPException(400, "agent_id required for new runs")

        agent_meta = registry.get_meta(nspace="agent", name=agent_id)
        if not agent_meta:
            raise HTTPException(404, f"Agent {agent_id} not found")

        # Pass files to agent input
        inputs = {
            "message": text,
            "files": incoming_files,
            "session_id": session_id,
            "user_meta": meta,
        }

        backing_name = agent_meta.get("backing", {}).get("name")
        if not backing_name:
            raise HTTPException(500, "Agent backing configuration missing")

        record = await rm.submit_run(
            graph_id=backing_name,
            inputs=inputs,
            session_id=session_id,
            identity=identity,
            agent_id=agent_id,
            origin="chat",  # or RunOrigin.chat if imported
            tags=["session:" + session_id, "agent:" + agent_id],
        )
        run_id = record.run_id

    return JSONResponse(
        {"ok": True, "resumed": resumed, "run_id": run_id, "files_processed": len(incoming_files)}
    )
