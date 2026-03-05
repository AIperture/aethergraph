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
from starlette.datastructures import FormData
from starlette.responses import JSONResponse

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.plugins.channel.mime_detect import detect_mime_for_path
from aethergraph.services.artifacts.facade import ArtifactFacade
from aethergraph.services.channel.attachments import (
    InputAttachment,
    attachment_to_dict,
    parse_input_attachments,
)
from aethergraph.services.channel.ingress import ChannelIngress, IncomingFile, IncomingMessage

router = APIRouter()


class RunChannelIncomingBody(BaseModel):
    text: str | None = None
    files: list[dict[str, Any]] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] | None = None


async def _save_upload_as_artifact(
    *,
    container: Any,
    upload: UploadFile,
    identity: RequestIdentity,
    session_id: str | None = None,
    run_id: str | None = None,
) -> Any:
    filename = upload.filename or "unknown"
    ext = ""
    if "." in filename:
        ext = f".{filename.split('.')[-1]}"

    tmp_path = await container.artifacts.plan_staging_path(
        planned_ext=f"_{uuid.uuid4().hex[:6]}{ext}"
    )

    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    scope = None
    if getattr(container, "scope_factory", None):
        scope = container.scope_factory.for_node(
            identity=identity,
            run_id=run_id,
            graph_id="chat",
            node_id="user_upload",
            session_id=session_id,
            app_id=None,
            tool_name="web.upload",
            tool_version="1.0.0",
        )

    artifact_facade = ArtifactFacade(
        run_id=run_id or f"session:{session_id}" if session_id else "webui",
        graph_id="chat",
        node_id="user_upload",
        tool_name="web.upload",
        tool_version="1.0.0",
        art_store=container.artifacts,
        art_index=container.artifact_index,
        scope=scope,
    )

    det = detect_mime_for_path(tmp_path)
    return await artifact_facade.save_file(
        path=tmp_path,
        kind="upload",
        suggested_uri=f"./uploads/{filename}"
        if not session_id
        else f"./sessions/{session_id}/uploads/{filename}",
        mime=det.detected_mime or upload.content_type or "application/octet-stream",
        labels={
            "source": "web_chat",
            "original_name": filename,
            "session_id": session_id or "",
            "run_id": run_id or "",
            "declared_content_type": upload.content_type or "",
            "detected_mime": det.detected_mime,
            "content_kind": det.content_kind,
            "detect_reason": det.reason,
        },
    )


def _attachment_to_incoming_file(a: InputAttachment) -> IncomingFile:
    return IncomingFile(
        id=a.artifact_id,
        name=a.name,
        mimetype=a.mimetype,
        size=a.size,
        url=a.url,
        uri=a.uri,
        extra={
            "source": a.source,
            "kind": a.kind,
            "labels": a.labels or {},
            "meta": a.meta or {},
        },
    )


async def _parse_run_incoming(
    *,
    request: Request,
) -> tuple[str, str | None, dict[str, Any], list[dict[str, Any]], list[UploadFile]]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form: FormData = await request.form()
        text = str(form.get("text") or "")
        choice = form.get("choice")
        meta: dict[str, Any] = {}
        attachments_raw: list[dict[str, Any]] = []
        meta_json = form.get("meta_json")
        if isinstance(meta_json, str) and meta_json:
            try:
                meta = json.loads(meta_json)
                if not isinstance(meta, dict):
                    raise HTTPException(400, "meta_json must be a JSON object")
            except json.JSONDecodeError as e:
                raise HTTPException(400, "Invalid meta JSON") from e
        attachments_json = form.get("attachments_json")
        if isinstance(attachments_json, str) and attachments_json:
            try:
                parsed = json.loads(attachments_json)
            except json.JSONDecodeError as e:
                raise HTTPException(400, "Invalid attachments JSON") from e
            if not isinstance(parsed, list):
                raise HTTPException(400, "attachments_json must be a JSON list")
            attachments_raw = parsed
        files: list[UploadFile] = []
        for _, v in form.multi_items():
            if isinstance(v, UploadFile) or (hasattr(v, "filename") and hasattr(v, "file")):
                files.append(v)
        return text, str(choice) if choice is not None else None, meta, attachments_raw, files

    try:
        body_obj = RunChannelIncomingBody.model_validate(await request.json())
    except Exception as e:
        raise HTTPException(400, "Invalid JSON body") from e
    attachments_raw = list(body_obj.attachments or [])
    for f in body_obj.files or []:
        if not isinstance(f, dict):
            continue
        artifact_id = f.get("artifact_id") or f.get("id")
        if artifact_id:
            attachments_raw.append(
                {
                    "kind": "artifact",
                    "source": f.get("source") or "context_ref",
                    "artifact_id": artifact_id,
                    "name": f.get("name"),
                    "mimetype": f.get("mimetype"),
                    "size": f.get("size"),
                    "uri": f.get("uri"),
                    "url": f.get("url"),
                    "labels": f.get("labels"),
                    "meta": f.get("meta"),
                }
            )

    return (
        body_obj.text or "",
        body_obj.choice,
        body_obj.meta or {},
        attachments_raw,
        [],
    )


async def _enrich_artifact_attachments(
    *,
    container: Any,
    attachments: list[InputAttachment],
) -> list[InputAttachment]:
    """
    Best-effort enrichment for artifact attachments using artifact store metadata.

    This helps downstream agents process context_ref attachments without re-querying
    artifact metadata for common fields like uri/name/mimetype.
    """
    if not attachments:
        return attachments

    artifact_index = getattr(container, "artifact_index", None)
    get_by_id = getattr(artifact_index, "get", None) if artifact_index is not None else None

    if get_by_id is None:
        for a in attachments:
            if a.kind == "artifact" and a.artifact_id and not a.url:
                a.url = f"/api/v1/artifacts/{a.artifact_id}/content"
        return attachments

    for a in attachments:
        if a.kind != "artifact" or not a.artifact_id:
            continue
        if not a.url:
            a.url = f"/api/v1/artifacts/{a.artifact_id}/content"
        try:
            art = await get_by_id(a.artifact_id)
        except Exception:
            continue
        if not art:
            continue

        if not a.name:
            a.name = getattr(art, "name", None) or getattr(art, "filename", None)
        if not a.mimetype:
            a.mimetype = getattr(art, "mime", None) or getattr(art, "mimetype", None)
        if a.size is None:
            a.size = getattr(art, "size", None)
        if not a.uri:
            a.uri = getattr(art, "uri", None)

        if not a.labels:
            art_labels = getattr(art, "labels", None)
            a.labels = dict(art_labels) if isinstance(art_labels, dict) else {}
    return attachments


@router.post("/runs/{run_id}/channel/incoming")
async def run_channel_incoming(
    run_id: str,
    request: Request,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> JSONResponse:
    try:
        container = request.app.state.container  # type: ignore
        ingress: ChannelIngress = container.channel_ingress
        event_log = container.eventlog

        text, choice, meta, attachments_raw, upload_files = await _parse_run_incoming(
            request=request
        )

        try:
            attachments = parse_input_attachments(attachments_raw)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        attachments = await _enrich_artifact_attachments(
            container=container,
            attachments=attachments,
        )

        incoming_files: list[IncomingFile] = [_attachment_to_incoming_file(a) for a in attachments]
        for upload in upload_files:
            artifact = await _save_upload_as_artifact(
                container=container,
                upload=upload,
                identity=identity,
                run_id=run_id,
            )
            attachment = InputAttachment(
                kind="artifact",
                source="upload",
                name=upload.filename,
                mimetype=artifact.mime,
                size=getattr(upload, "size", None),
                artifact_id=artifact.artifact_id,
                url=f"/api/v1/artifacts/{artifact.artifact_id}/content",
                uri=artifact.uri,
                labels={
                    "session_id": "",
                    "run_id": run_id,
                    "content_kind": artifact.labels.get("content_kind"),
                },
            )
            attachments.append(attachment)
            incoming_files.append(_attachment_to_incoming_file(attachment))

        if text:
            now_ts = datetime.now(timezone.utc).timestamp()
            await event_log.append(
                {
                    "id": str(uuid4()),
                    "ts": now_ts,
                    "scope_id": run_id,
                    "kind": "run_channel",
                    "payload": {
                        "type": "user.message",
                        "text": text,
                        "buttons": [],
                        "file": None,
                        "files": [dataclasses.asdict(f) for f in incoming_files],
                        "attachments": [attachment_to_dict(a) for a in attachments],
                        "meta": {
                            **meta,
                            "direction": "inbound",
                            "role": "user",
                        },
                    },
                }
            )

        resumed = await ingress.handle(
            IncomingMessage(
                scheme="ui",
                channel_id=f"run/{run_id}",
                thread_id=None,
                text=text,
                files=incoming_files or None,
                attachments=attachments or None,
                choice=choice,
                meta=meta,
            )
        )

        return JSONResponse({"ok": True, "resumed": resumed, "files_processed": len(upload_files)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/sessions/{session_id}/chat/incoming")
async def session_chat_incoming(
    session_id: str,
    text: str = Form(""),
    agent_id: str | None = Form(None),  # noqa: B008
    meta_json: str | None = Form(None),  # noqa: B008
    attachments_json: str | None = Form(None),  # noqa: B008
    files: list[UploadFile] = File(default=[]),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
):
    container = current_services()
    ingress = container.channel_ingress
    registry = container.registry
    rm = container.run_manager
    event_log = container.eventlog

    meta: dict[str, Any] = {}
    if meta_json:
        try:
            parsed_meta = json.loads(meta_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid meta JSON") from e
        if not isinstance(parsed_meta, dict):
            raise HTTPException(400, "meta_json must be a JSON object")
        meta = parsed_meta

    raw_attachments: list[dict[str, Any]] = []
    if attachments_json:
        try:
            parsed_attachments = json.loads(attachments_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid attachments JSON") from e
        if not isinstance(parsed_attachments, list):
            raise HTTPException(400, "attachments_json must be a JSON list")
        raw_attachments = parsed_attachments

    try:
        attachments = parse_input_attachments(raw_attachments)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    attachments = await _enrich_artifact_attachments(
        container=container,
        attachments=attachments,
    )

    incoming_files: list[IncomingFile] = [_attachment_to_incoming_file(a) for a in attachments]
    for upload in files:
        artifact = await _save_upload_as_artifact(
            container=container,
            upload=upload,
            identity=identity,
            session_id=session_id,
        )
        attachment = InputAttachment(
            kind="artifact",
            source="upload",
            name=upload.filename,
            mimetype=artifact.mime,
            size=getattr(upload, "size", None),
            artifact_id=artifact.artifact_id,
            url=f"/api/v1/artifacts/{artifact.artifact_id}/content",
            uri=artifact.uri,
            labels={
                "session_id": session_id,
                "content_kind": artifact.labels.get("content_kind"),
            },
        )
        attachments.append(attachment)
        incoming_files.append(_attachment_to_incoming_file(attachment))

    if text or incoming_files:
        now_ts = datetime.now(timezone.utc).timestamp()
        files_payload = []
        for f in incoming_files:
            payload_file = dataclasses.asdict(f)
            if payload_file.get("id") and not payload_file.get("artifact_id"):
                payload_file["artifact_id"] = payload_file["id"]
            files_payload.append(payload_file)

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
                    "attachments": [attachment_to_dict(a) for a in attachments],
                    "meta": {
                        **meta,
                        "direction": "inbound",
                        "role": "user",
                        "attachments": [attachment_to_dict(a) for a in attachments],
                    },
                },
            }
        )

    resumed = await ingress.handle(
        IncomingMessage(
            scheme="ui",
            channel_id=f"session/{session_id}",
            thread_id=None,
            text=text,
            files=incoming_files or None,
            attachments=attachments or None,
            meta=meta,
        )
    )

    run_id: str | None = None
    if not resumed:
        if agent_id is None:
            raise HTTPException(
                status_code=400,
                detail="agent_id is required when no continuation is resumed",
            )

        agent_meta = registry.get_meta(nspace="agent", name=agent_id)
        if not agent_meta:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {agent_id}",
            )

        run_vis = RunVisibility(agent_meta.get("run_visibility", RunVisibility.inline.value))
        run_imp = RunImportance(agent_meta.get("run_importance", RunImportance.ephemeral.value))

        backing = agent_meta.get("backing", {})
        if backing.get("type") != "graphfn":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported agent backing type: {backing.get('type')}. Only 'graphfn' is supported in v1.",
            )

        graph_id = backing["name"]
        inputs = {
            "message": text,
            "attachments": [attachment_to_dict(a) for a in attachments],
            "session_id": session_id,
            "user_meta": meta or {},
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
            app_id=agent_meta.get("app_id"),
            tags=["session:" + session_id, "agent:" + agent_id],
        )
        run_id = record.run_id

    return JSONResponse(
        {
            "ok": True,
            "resumed": resumed,
            "run_id": run_id,
            "files_processed": len(files),
        }
    )
