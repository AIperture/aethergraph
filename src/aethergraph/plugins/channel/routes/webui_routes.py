from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.datastructures import FormData
from starlette.responses import JSONResponse

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.plugins.channel.utils.turn_dispatch import dispatch_channel_turn_run
from aethergraph.services.channel.ingress import (
    ChannelIngress,
    IncomingMessage,
    IngressPersistenceScope,
)
from aethergraph.services.channel.resources import (
    ArtifactIngressScope,
    InputResource,
    InputResourceNormalizer,
    ResourceEnricher,
    ResourceSet,
    ResourceStager,
)

router = APIRouter()


class RunChannelIncomingBody(BaseModel):
    text: str | None = None
    files: list[dict[str, Any]] | None = None
    choice: str | None = None
    meta: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] | None = None
    context_refs: list[dict[str, Any]] | None = None


_NORMALIZER = InputResourceNormalizer()


def _parse_resource_json(raw: Any, *, field_name: str, source: str) -> ResourceSet:
    if raw is None:
        return ResourceSet()
    if not isinstance(raw, list):
        raise HTTPException(400, f"{field_name} must be a JSON list")

    resources = ResourceSet()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HTTPException(400, f"{field_name}[{i}] must be an object")
        resources.add(_NORMALIZER.from_dict(item, source=source))
    return resources


def _read_json_list(raw: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not isinstance(raw, str):
        raise HTTPException(400, f"{field_name} must be a JSON list")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid {field_name} JSON") from e
    if not isinstance(parsed, list):
        raise HTTPException(400, f"{field_name} must be a JSON list")
    return parsed


async def _stage_upload_resource(
    *,
    container: Any,
    upload: UploadFile,
    identity: RequestIdentity,
    session_id: str | None = None,
    run_id: str | None = None,
) -> InputResource:
    filename = upload.filename or "unknown"
    data = await upload.read()
    scope_run_id = None if session_id else run_id
    return await ResourceStager(container=container, identity=identity).stage_bytes(
        data,
        name=filename,
        mime=upload.content_type or "application/octet-stream",
        scope=ArtifactIngressScope(
            source="webui",
            session_id=session_id,
            run_id=scope_run_id,
            channel_key=f"ui:session/{session_id}" if session_id else f"ui:run/{run_id}",
            conversation_id=f"ui:session/{session_id}" if session_id else f"ui:run/{run_id}",
            graph_id="chat",
            node_id="user_upload",
            tool_name="web.upload",
            tool_version="1.0.0",
        ),
        labels={
            "original_name": filename,
            "declared_content_type": upload.content_type or "",
        },
        meta={"upload_size": len(data)},
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
        resources_raw: list[dict[str, Any]] = []
        meta_json = form.get("meta_json")
        if isinstance(meta_json, str) and meta_json:
            try:
                meta = json.loads(meta_json)
                if not isinstance(meta, dict):
                    raise HTTPException(400, "meta_json must be a JSON object")
            except json.JSONDecodeError as e:
                raise HTTPException(400, "Invalid meta JSON") from e
        resources_raw.extend(
            _read_json_list(form.get("attachments_json"), field_name="attachments_json")
        )
        resources_raw.extend(
            _read_json_list(form.get("context_refs_json"), field_name="context_refs_json")
        )
        files: list[UploadFile] = []
        for _, v in form.multi_items():
            if isinstance(v, UploadFile) or (hasattr(v, "filename") and hasattr(v, "file")):
                files.append(v)
        return text, str(choice) if choice is not None else None, meta, resources_raw, files

    try:
        body_obj = RunChannelIncomingBody.model_validate(await request.json())
    except Exception as e:
        raise HTTPException(400, "Invalid JSON body") from e
    resources_raw = list(body_obj.attachments or [])
    resources_raw.extend(body_obj.context_refs or [])
    resources_raw.extend(body_obj.files or [])
    for f in resources_raw:
        if not isinstance(f, dict):
            raise HTTPException(400, "resource entries must be objects")

    return (
        body_obj.text or "",
        body_obj.choice,
        body_obj.meta or {},
        resources_raw,
        [],
    )


async def _enrich_resources(
    *,
    container: Any,
    resources: ResourceSet,
) -> ResourceSet:
    return await ResourceEnricher(container=container).enrich(resources)


@router.post("/runs/{run_id}/channel/incoming")
async def run_channel_incoming(
    run_id: str,
    request: Request,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> JSONResponse:
    try:
        container = request.app.state.container  # type: ignore
        ingress: ChannelIngress = container.channel_ingress

        text, choice, meta, resources_raw, upload_files = await _parse_run_incoming(request=request)

        resources = _parse_resource_json(resources_raw, field_name="resources", source="webui")
        resources = await _enrich_resources(
            container=container,
            resources=resources,
        )

        for upload in upload_files:
            resources.add(
                await _stage_upload_resource(
                    container=container,
                    upload=upload,
                    identity=identity,
                    run_id=run_id,
                )
            )
        resources.dedupe()

        result = await ingress.accept(
            IncomingMessage(
                scheme="ui",
                channel_id=f"run/{run_id}",
                thread_id=None,
                text=text,
                attachments=list(resources.resources) or None,
                choice=choice,
                conversation_id=f"ui:run/{run_id}",
                meta=meta,
            ),
            persistence_scope=IngressPersistenceScope(
                scope_id=run_id,
                kind="run_channel",
                user_id=identity.user_id,
                org_id=identity.org_id,
            ),
            source_meta={"identity_mode": identity.mode},
        )

        return JSONResponse(
            {
                "ok": True,
                "resumed": result.resumed,
                "files_processed": len(upload_files),
            }
        )
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
    context_refs_json: str | None = Form(None),  # noqa: B008
    files: list[UploadFile] = File(default=[]),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
):
    container = current_services()
    ingress = container.channel_ingress
    registry = scoped_registry(identity)

    meta: dict[str, Any] = {}
    if meta_json:
        try:
            parsed_meta = json.loads(meta_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid meta JSON") from e
        if not isinstance(parsed_meta, dict):
            raise HTTPException(400, "meta_json must be a JSON object")
        meta = parsed_meta

    resources = _parse_resource_json(
        [
            *_read_json_list(attachments_json, field_name="attachments_json"),
            *_read_json_list(context_refs_json, field_name="context_refs_json"),
        ],
        field_name="resources",
        source="webui",
    )
    resources = await _enrich_resources(
        container=container,
        resources=resources,
    )

    for upload in files:
        resources.add(
            await _stage_upload_resource(
                container=container,
                upload=upload,
                identity=identity,
                session_id=session_id,
            )
        )
    resources.dedupe()
    display_files = resources.to_display_files()
    attachment_payloads = resources.to_attachment_dicts()

    result = await ingress.accept(
        IncomingMessage(
            scheme="ui",
            channel_id=f"session/{session_id}",
            thread_id=None,
            text=text,
            attachments=list(resources.resources) or None,
            conversation_id=f"ui:session/{session_id}",
            meta={
                **meta,
                "_drop_stale_continuation": True,
            },
        ),
        persistence_scope=IngressPersistenceScope(
            scope_id=session_id,
            kind="session_chat",
            user_id=identity.user_id,
            org_id=identity.org_id,
        ),
        source_meta={"identity_mode": identity.mode},
    )
    resumed = result.resumed

    run_id: str | None = None
    if not resumed:
        if agent_id is None:
            raise HTTPException(
                status_code=400,
                detail="agent_id is required when no continuation is resumed",
            )

        _ = registry  # kept to avoid changing route wiring; dispatch helper resolves through the same facade
        run_id = await dispatch_channel_turn_run(
            container=container,
            identity=identity,
            agent_id=agent_id,
            text=text,
            attachments=list(resources.resources),
            session_id=session_id,
            user_meta={
                **(meta or {}),
                "attachments": attachment_payloads,
                "conversation_id": f"ui:session/{session_id}",
            },
            tags=["session:" + session_id, "agent:" + agent_id],
        )

    return JSONResponse(
        {
            "ok": True,
            "resumed": resumed,
            "run_id": run_id,
            "files_processed": len(files),
            # Return the persisted display files (carrying artifact_id) so the
            # client can swap its optimistic blob previews for durable artifact
            # URLs immediately, without waiting for the websocket echo/refresh.
            "files": display_files,
        }
    )
