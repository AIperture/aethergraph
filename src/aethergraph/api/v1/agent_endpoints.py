"""Public Agent Endpoint API over canonical ingress and semantic events."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from aethergraph.contracts.integration import (
    ExternalIdentity,
    IngressAttachment,
    IngressChoice,
    IngressEnvelope,
    IntegrationKind,
    OriginAddress,
)
from aethergraph.core.runtime.run_types import SessionKind
from aethergraph.services.integration import VerifiedAttachment, VerifiedIntegrationContext

from .deps import RequestIdentity, artifact_belongs_to_identity, get_identity

router = APIRouter(prefix="/agent-endpoints", tags=["agent-endpoints"])


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointSessionCreate(_ClosedModel):
    idempotency_key: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=512)


class EndpointSessionView(_ClosedModel):
    session_id: str
    endpoint_id: str


class EndpointChoice(_ClosedModel):
    interaction_id: str = Field(min_length=1, max_length=255)
    option_ids: tuple[str, ...]


class EndpointArtifactInput(_ClosedModel):
    artifact_id: str = Field(min_length=1, max_length=255)
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)


class EndpointIngressBody(_ClosedModel):
    session_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=255)
    text: str | None = Field(default=None, max_length=1_000_000)
    choice: EndpointChoice | None = None
    attachments: tuple[EndpointArtifactInput, ...] = ()
    structured_input: dict[str, Any] | None = None


class EndpointCancelBody(_ClosedModel):
    turn_id: str = Field(min_length=1, max_length=255)


def _host(request: Request):
    container = request.app.state.container
    if container.integration_ingress is None or container.host_manifest is None:
        raise HTTPException(status_code=503, detail="AG Host integration ingress is not installed.")
    return container


def _principal(identity: RequestIdentity) -> tuple[str, str]:
    if identity.user_id and identity.org_id:
        return identity.org_id, identity.user_id
    if identity.is_local:
        return "local", identity.user_id or "local"
    raise HTTPException(status_code=401, detail="Authenticated user and tenant identity required.")


def _route(container, endpoint_id: str):
    matches = [
        route
        for route in container.host_manifest.integration_routes
        if route.endpoint_id == endpoint_id
        and route.integration_kind in {IntegrationKind.AG_UI, IntegrationKind.AGENT_ENDPOINT}
    ]
    if len(matches) != 1 or not matches[0].enabled:
        raise HTTPException(status_code=404, detail="Agent endpoint not found.")
    return matches[0]


def _external_identity(
    *,
    identity: RequestIdentity,
    session_id: str,
) -> ExternalIdentity:
    tenant_id, user_id = _principal(identity)
    return ExternalIdentity(
        tenant_id=tenant_id,
        conversation_id=session_id,
        user_id=user_id,
    )


def _verified(
    *,
    route,
    identity: RequestIdentity,
    attachments: tuple[VerifiedAttachment, ...] = (),
) -> VerifiedIntegrationContext:
    tenant_id, _ = _principal(identity)
    return VerifiedIntegrationContext(
        integration_id=route.integration_id,
        integration_kind=route.integration_kind,
        external_tenant_id=tenant_id,
        attachments=attachments,
        request_identity=identity,
    )


async def _binding(container, *, route, external_identity: ExternalIdentity):
    binding = await container.integration_ingress.binding_store.get(
        route=route,
        external_identity=external_identity,
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="Endpoint session not found.")
    return binding


@router.post("/{endpoint_id}/sessions", response_model=EndpointSessionView)
async def create_endpoint_session(
    endpoint_id: str,
    body: EndpointSessionCreate,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> EndpointSessionView:
    """Create or replay one authenticated public endpoint session.

    The idempotency key deterministically identifies the public conversation;
    the same identity is used by the public endpoint and AG session store.

    Examples:
        Create a bespoke application session:
        ```python
        POST /api/v1/agent-endpoints/support/sessions
        {"idempotency_key": "browser-session-1"}
        ```

        Replay the same creation safely:
        ```python
        POST /api/v1/agent-endpoints/support/sessions
        {"idempotency_key": "browser-session-1"}
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        body: Closed idempotent session-creation command.
        request: FastAPI request carrying the installed AG Host.
        identity: Authenticated endpoint principal.

    Returns:
        EndpointSessionView: Public session and endpoint identities.

    Notes:
        Clients never select an AG agent, graph, or internal session identifier.
    """
    container = _host(request)
    route = _route(container, endpoint_id)
    tenant_id, user_id = _principal(identity)
    digest = sha256(
        f"{container.host_manifest.deployment_id}:{endpoint_id}:{tenant_id}:{user_id}:"
        f"{body.idempotency_key}".encode()
    ).hexdigest()
    session_id = f"endpoint-session-{digest[:32]}"
    external = _external_identity(identity=identity, session_id=session_id)
    verified = _verified(route=route, identity=identity)
    probe = IngressEnvelope(
        integration_id=route.integration_id,
        endpoint_id=endpoint_id,
        external_identity=external,
        external_event_id=f"session-create-{digest[:24]}",
        idempotency_key=f"session-create-{digest[:24]}",
        received_at=datetime.now(UTC),
        structured_input={"operation": "session.create"},
        origin_address=OriginAddress(
            channel_key=f"endpoint:{endpoint_id}:session/{session_id}",
            capability_profile_id="agent-endpoint-v1",
        ),
    )
    resolved = container.integration_ingress.route_resolver.resolve(
        verified=verified,
        envelope=probe,
    )
    existing_binding = await container.integration_ingress.binding_store.get(
        route=resolved,
        external_identity=external,
    )
    if existing_binding is not None and existing_binding.ag_session_id != session_id:
        raise HTTPException(
            status_code=409,
            detail="Endpoint session binding conflicts with the canonical session identity.",
        )
    await container.session_store.create(
        session_id=session_id,
        kind=SessionKind.chat,
        user_id=user_id,
        org_id=tenant_id,
        title=body.title,
        source=("ag_ui" if route.integration_kind is IntegrationKind.AG_UI else "agent_endpoint"),
        external_ref=f"agent-endpoint:{endpoint_id}",
    )
    binding = await container.integration_ingress.binding_store.get_or_create(
        route=resolved,
        external_identity=external,
        build_id=container.host_manifest.build_id,
        binding_id=f"binding-{digest[:32]}",
        ag_session_id=session_id,
        now=probe.received_at,
    )
    if binding.binding.ag_session_id != session_id:
        raise HTTPException(
            status_code=409,
            detail="Endpoint session binding conflicts with the canonical session identity.",
        )
    return EndpointSessionView(session_id=session_id, endpoint_id=endpoint_id)


async def _parse_ingress(request: Request) -> tuple[EndpointIngressBody, tuple[UploadFile, ...]]:
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        return EndpointIngressBody.model_validate(await request.json()), ()
    form = await request.form()
    choice = json.loads(str(form["choice_json"])) if form.get("choice_json") else None
    artifacts = json.loads(str(form["attachments_json"])) if form.get("attachments_json") else []
    structured = (
        json.loads(str(form["structured_input_json"]))
        if form.get("structured_input_json")
        else None
    )
    body = EndpointIngressBody.model_validate(
        {
            "session_id": form.get("session_id"),
            "idempotency_key": form.get("idempotency_key"),
            "text": form.get("text"),
            "choice": choice,
            "attachments": artifacts,
            "structured_input": structured,
        }
    )
    uploads = tuple(value for _, value in form.multi_items() if isinstance(value, UploadFile))
    return body, uploads


@router.post("/{endpoint_id}/ingress")
async def endpoint_ingress(
    endpoint_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
):
    """Accept JSON or multipart input through the canonical coordinator.

    Artifact references and uploaded bytes converge on shared ResourceIngress;
    the endpoint never stages files or constructs agent payloads.

    Examples:
        Send JSON text:
        ```python
        POST /api/v1/agent-endpoints/support/ingress
        {"session_id": "...", "idempotency_key": "turn-1", "text": "Hello"}
        ```

        Submit an exact interaction:
        ```python
        {"session_id": "...", "idempotency_key": "choice-1",
         "choice": {"interaction_id": "interaction-1", "option_ids": ["ship"]}}
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        request: JSON or multipart FastAPI request.
        identity: Authenticated endpoint principal.

    Returns:
        IngressReceipt: Canonical accepted, duplicate, or rejected receipt.

    Notes:
        Request bodies cannot supply agents, graphs, run configuration, paths,
        channel keys, or continuation tokens.
    """
    container = _host(request)
    route = _route(container, endpoint_id)
    try:
        body, uploads = await _parse_ingress(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid endpoint ingress body.") from exc
    external = _external_identity(identity=identity, session_id=body.session_id)
    await _binding(container, route=route, external_identity=external)
    declared = [
        IngressAttachment(
            attachment_id=item.artifact_id,
            source_kind="artifact",
            source_id=item.artifact_id,
            filename=item.filename,
            content_type=item.content_type,
            size_bytes=item.size_bytes,
        )
        for item in body.attachments
    ]
    verified_files: list[VerifiedAttachment] = []
    for index, upload in enumerate(uploads):
        data = await upload.read()
        attachment_id = f"upload-{index}"
        declared.append(
            IngressAttachment(
                attachment_id=attachment_id,
                source_kind="provider_file",
                source_id=attachment_id,
                filename=upload.filename or f"upload-{index}",
                content_type=upload.content_type or "application/octet-stream",
                size_bytes=len(data),
            )
        )
        verified_files.append(VerifiedAttachment(attachment_id=attachment_id, data=data))
    envelope = IngressEnvelope(
        integration_id=route.integration_id,
        endpoint_id=endpoint_id,
        external_identity=external,
        external_event_id=body.idempotency_key,
        idempotency_key=body.idempotency_key,
        received_at=datetime.now(UTC),
        text=body.text,
        choice=(
            IngressChoice(
                interaction_id=body.choice.interaction_id,
                option_ids=body.choice.option_ids,
            )
            if body.choice
            else None
        ),
        attachments=tuple(declared),
        structured_input=body.structured_input,
        origin_address=OriginAddress(
            channel_key=f"endpoint:{endpoint_id}:session/{body.session_id}",
            capability_profile_id="agent-endpoint-v1",
        ),
    )
    return await container.integration_ingress.accept(
        verified=_verified(
            route=route,
            identity=identity,
            attachments=tuple(verified_files),
        ),
        envelope=envelope,
    )


async def _session_binding(request, endpoint_id, session_id, identity):
    container = _host(request)
    route = _route(container, endpoint_id)
    external = _external_identity(identity=identity, session_id=session_id)
    return container, route, await _binding(container, route=route, external_identity=external)


@router.get("/{endpoint_id}/sessions/{session_id}/events")
async def endpoint_session_events(
    endpoint_id: str,
    session_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    after_cursor: int | None = Query(default=None, ge=0),  # noqa: B008
    limit: int = Query(default=100, ge=1, le=500),  # noqa: B008
) -> dict[str, Any]:
    """Read cursor-ordered semantic history for one endpoint session.

    Examples:
        Read initial history:
        ```python
        GET /api/v1/agent-endpoints/support/sessions/s1/events
        ```

        Resume after a cursor:
        ```python
        GET /api/v1/agent-endpoints/support/sessions/s1/events?after_cursor=42
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.
        after_cursor: Optional exclusive durable cursor.
        limit: Maximum number of semantic events.

    Returns:
        dict[str, Any]: Ordered events and the last returned cursor.

    Notes:
        History and live streaming use the same semantic store and cursor contract.
    """
    container, _, binding = await _session_binding(request, endpoint_id, session_id, identity)
    rows = await container.semantic_events.list_session(
        deployment_id=container.host_manifest.deployment_id,
        session_id=binding.ag_session_id,
        after_cursor=after_cursor,
        limit=limit,
    )
    return {
        "events": [
            {"cursor": row.cursor, "event": row.event.model_dump(mode="json")} for row in rows
        ],
        "next_cursor": rows[-1].cursor if rows else after_cursor,
    }


@router.get("/{endpoint_id}/sessions/{session_id}/stream")
async def endpoint_session_stream(
    endpoint_id: str,
    session_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    after_cursor: int | None = Query(default=None, ge=0),  # noqa: B008
) -> StreamingResponse:
    """Stream semantic events from the same reconnect cursor used by history.

    Examples:
        Open a fresh event stream:
        ```python
        GET /api/v1/agent-endpoints/support/sessions/s1/stream
        ```

        Reconnect without replaying delivered events:
        ```python
        GET /api/v1/agent-endpoints/support/sessions/s1/stream?after_cursor=42
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        request: FastAPI request used for disconnect detection.
        identity: Authenticated endpoint principal.
        after_cursor: Optional exclusive durable cursor.

    Returns:
        StreamingResponse: Server-sent semantic event stream.

    Notes:
        The stream polls only the canonical semantic store; no EventHub projection exists.
    """
    container, _, binding = await _session_binding(request, endpoint_id, session_id, identity)

    async def generate():
        cursor = after_cursor
        while not await request.is_disconnected():
            rows = await container.semantic_events.list_session(
                deployment_id=container.host_manifest.deployment_id,
                session_id=binding.ag_session_id,
                after_cursor=cursor,
                limit=100,
            )
            if rows:
                for row in rows:
                    cursor = row.cursor
                    data = json.dumps(
                        {"cursor": row.cursor, "event": row.event.model_dump(mode="json")}
                    )
                    yield f"id: {row.cursor}\nevent: semantic\ndata: {data}\n\n"
            else:
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/{endpoint_id}/sessions/{session_id}/cancel")
async def cancel_endpoint_turn(
    endpoint_id: str,
    session_id: str,
    body: EndpointCancelBody,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> dict[str, str]:
    """Cancel one exact turn after validating endpoint-session ownership.

    Examples:
        Cancel an active turn:
        ```python
        POST /api/v1/agent-endpoints/support/sessions/s1/cancel
        {"turn_id": "run-1"}
        ```

        Retry the exact cancellation:
        ```python
        POST /api/v1/agent-endpoints/support/sessions/s1/cancel
        {"turn_id": "run-1"}
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        body: Exact turn cancellation command.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.

    Returns:
        dict[str, str]: Turn identity and cancellation-requested status.

    Notes:
        The client cannot cancel a turn belonging to another bound session.
    """
    container, _, binding = await _session_binding(request, endpoint_id, session_id, identity)
    record = await container.run_store.get(body.turn_id)
    if record is None or record.session_id != binding.ag_session_id:
        raise HTTPException(status_code=404, detail="Turn not found.")
    await container.run_manager.cancel_run(body.turn_id)
    return {"turn_id": body.turn_id, "status": "cancellation_requested"}


@router.get("/{endpoint_id}/artifacts/{artifact_id}")
async def endpoint_artifact(
    endpoint_id: str,
    artifact_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> Response:
    """Download one endpoint-scoped artifact after identity validation.

    Examples:
        Download a generated file:
        ```python
        GET /api/v1/agent-endpoints/support/artifacts/artifact-1
        ```

        Use the response content type:
        ```python
        response = client.get("/api/v1/agent-endpoints/support/artifacts/artifact-1")
        ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        artifact_id: Exact artifact identity.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.

    Returns:
        Response: Artifact bytes with recorded content type.

    Notes:
        Unknown, unauthorized, and cross-route artifacts all return not found.
    """
    container = _host(request)
    route = _route(container, endpoint_id)
    artifact = await container.artifact_index.get(artifact_id)
    labels = artifact.labels if artifact is not None else {}
    if (
        artifact is None
        or not artifact_belongs_to_identity(identity, artifact)
        or labels.get("route_id") != route.route_id
    ):
        raise HTTPException(status_code=404, detail="Artifact not found.")
    data = await container.artifacts.load_artifact_bytes(artifact.uri)
    return Response(
        content=data,
        media_type=artifact.mime or "application/octet-stream",
        headers={"X-AetherGraph-Artifact-Id": artifact.artifact_id},
    )
