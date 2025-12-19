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
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.artifacts.facade import ArtifactFacade
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
    agent_id: str | None = None
    context_refs: list[dict[str, Any]] | None = None


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


async def _save_upload_as_artifact_deprecated(
    container, upload: UploadFile, session_id: str, identity: RequestIdentity
) -> str:
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


async def _save_upload_as_artifact(
    container: Any,
    upload: UploadFile,
    session_id: str,
    identity: RequestIdentity,
) -> str:
    """
    Streams upload to disk, saves as session-scoped artifact, returns URI.
    Artifacts created here will appear under scope_id = session_id.
    """
    filename = upload.filename or "unknown"
    ext = ""
    if "." in filename:
        ext = f".{filename.split('.')[-1]}"

    # 1. Stage to a temp path
    tmp_path = await container.artifacts.plan_staging_path(
        planned_ext=f"_{uuid.uuid4().hex[:6]}{ext}"
    )

    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    # 2. Build a Scope for this session upload
    scope = None
    if getattr(container, "scope_factory", None):
        scope = container.scope_factory.for_node(
            identity=identity,
            run_id=None,
            graph_id="chat",
            node_id="user_upload",
            session_id=session_id,
            app_id=None,
            tool_name="web.upload",
            tool_version="1.0.0",
        )

    # 3. Use ArtifactFacade so index gets scope_id = session_id
    artifact_facade = ArtifactFacade(
        run_id=f"session:{session_id}",
        graph_id="chat",
        node_id="user_upload",
        tool_name="web.upload",
        tool_version="1.0.0",
        store=container.artifacts,
        index=container.artifact_index,
        scope=scope,
    )

    artifact = await artifact_facade.save_file(
        path=tmp_path,
        kind="upload",
        suggested_uri=f"./sessions/{session_id}/uploads/{filename}",
        labels={
            "source": "web_chat",
            "original_name": filename,
            "session_id": session_id,
            "content_type": upload.content_type or "",
        },
    )

    # Return URI (or local path fallback)
    return getattr(artifact, "uri", None) or getattr(artifact, "path", None)


@router.post("/sessions/{session_id}/chat/incoming")
async def session_chat_incoming(
    session_id: str,
    request: Request,
    # Form fields
    text: str = Form(""),
    agent_id: str | None = Form(None),  # noqa: B008
    meta_json: str | None = Form(None),  # noqa: B008
    context_refs_json: str | None = Form(None),  # 🔑 new
    # Files
    files: list[UploadFile] = File(default=[]),  # noqa: B008
    # Context
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
):
    container = current_services()
    ingress = container.channel_ingress
    registry = container.registry
    rm = container.run_manager
    event_log = container.eventlog

    # 1. Parse meta
    meta: dict[str, Any] = {}
    if meta_json:
        try:
            meta = json.loads(meta_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid meta JSON") from e

    # 2. Parse context_refs (JSON list)
    context_refs: list[dict[str, Any]] = []
    if context_refs_json:
        try:
            raw = json.loads(context_refs_json)
            if isinstance(raw, list):
                context_refs = raw
            else:
                raise HTTPException(400, "context_refs_json must be a JSON list")
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid context_refs JSON") from e

    # 3. Process files -> IncomingFile (and save as artifacts)
    incoming_files: list[IncomingFile] = []
    for upload in files:
        uri = await _save_upload_as_artifact(container, upload, session_id, identity)
        incoming_files.append(
            IncomingFile(
                id=str(uuid.uuid4()),
                name=upload.filename,
                mimetype=upload.content_type,
                size=getattr(upload, "size", None),
                url=None,
                uri=uri,
                extra={
                    "source": "web_upload",
                    "session_id": session_id,
                },
            )
        )

    # 4. Log event (with files + context_refs in meta)
    if text or incoming_files:
        now_ts = datetime.now(timezone.utc).timestamp()
        files_payload = [dataclasses.asdict(f) for f in incoming_files]

        log_meta = {
            **meta,
            "direction": "inbound",
            "role": "user",
        }
        if context_refs:
            log_meta["context_refs"] = context_refs

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
                    "meta": log_meta,
                },
            }
        )

    # 5. Let ChannelIngress handle / resume continuations
    msg_meta = dict(meta)
    if context_refs:
        msg_meta["context_refs"] = context_refs

    resumed = await ingress.handle(
        IncomingMessage(
            scheme="ui",
            channel_id=f"session/{session_id}",
            thread_id=None,
            text=text,
            files=incoming_files or None,
            meta=msg_meta,
        )
    )

    # 6. Spawn run if nothing was resumed
    run_id: str | None = None
    if not resumed:
        if agent_id is None:
            # for v1 it is fine to require frontend to specify agent_id
            # later we can derive default agent per session
            raise HTTPException(
                status_code=400,
                detail="agent_id is required when no continuation is resumed",
            )

        # Resolve agent meta -> backing graph
        agent_meta = registry.get_meta(nspace="agent", name=agent_id)
        if not agent_meta:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {agent_id}",
            )

        run_vis_str = agent_meta.get("run_visibility", RunVisibility.inline.value)  # default inline
        run_imp_str = agent_meta.get(
            "run_importance", RunImportance.ephemeral.value
        )  # default ephemeral
        run_vis = RunVisibility(run_vis_str)
        run_imp = RunImportance(run_imp_str)

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
            "files": incoming_files,
            "session_id": session_id,  # for convenience, we can derive session inside graph too
            "user_meta": meta or {},  # optional user meta
            "context_refs": context_refs or [],  # optional context references
        }

        record = await rm.submit_run(
            graph_id=graph_id,
            inputs=inputs,
            session_id=session_id,
            identity=identity,
            origin=RunOrigin.chat,
            visibility=run_vis,
            importance=run_imp,
            agent_id=agent_id,
            app_id=agent_meta.get("app_id"),  # optional, if you attach this
            tags=["session:" + session_id, "agent:" + agent_id],
        )
        run_id = record.run_id

    return JSONResponse(
        {
            "ok": True,
            "resumed": resumed,
            "run_id": run_id,
            "files_processed": len(incoming_files),
        }
    )
