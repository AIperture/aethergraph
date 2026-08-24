"""Public Agent Endpoint API over canonical ingress and semantic events."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from ipaddress import ip_address
import json
from typing import Annotated, Any, Literal, Never

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile as StarletteUploadFile

from aethergraph.contracts.integration import (
    ExternalIdentity,
    IngressAttachment,
    IngressChoice,
    IngressEnvelope,
    IntegrationKind,
    OriginAddress,
)
from aethergraph.core.runtime.run_types import SessionKind
from aethergraph.services.host.endpoint_credentials import ENDPOINT_COOKIE_NAME
from aethergraph.services.integration import (
    ResourceIngressError,
    ResourceIngressPolicy,
    VerifiedAttachment,
    VerifiedIntegrationContext,
    read_bounded_attachment_bytes,
)
from aethergraph.storage.contracts import StorageScope

from .deps import RequestIdentity, artifact_belongs_to_identity, get_identity
from .pagination import decode_cursor, encode_cursor
from .schemas.session import Session, SessionListResponse

router = APIRouter(prefix="/agent-endpoints", tags=["agent-endpoints"])


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointSessionCreate(_ClosedModel):
    idempotency_key: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=512)


class EndpointSessionView(_ClosedModel):
    session_id: str
    endpoint_id: str


class EndpointDescriptor(_ClosedModel):
    endpoint_id: str
    entry_agent_id: str


class EndpointSessionUpdate(_ClosedModel):
    title: str | None = Field(default=None, max_length=512)


class EndpointChoice(_ClosedModel):
    interaction_id: str = Field(min_length=1, max_length=255)
    option_ids: tuple[str, ...]


class EndpointArtifactInput(_ClosedModel):
    artifact_id: str = Field(min_length=1, max_length=255)
    filename: str | None = Field(default=None, min_length=1, max_length=512)
    content_type: str | None = Field(default=None, min_length=1, max_length=255)
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


def _credential_registry(request: Request):
    registry = getattr(request.app.state, "endpoint_credentials", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Endpoint credentials are unavailable.")
    return registry


def require_endpoint_credential(endpoint_id: str, request: Request) -> None:
    """Require the bounded cookie issued for one exact Agent Endpoint.

    Examples:
        Protect a route with the shared dependency:
        ```python
        credential: Annotated[None, Depends(require_endpoint_credential)]
        ```

    Args:
        endpoint_id: Exact endpoint identity from the public request path.
        request: FastAPI request carrying the Host credential registry.

    Returns:
        None: Returns only after endpoint-scoped credential validation.

    Notes:
        Generic AG local identity is not an authentication fallback for Host endpoints.
    """

    if getattr(request.app.state, "development_ui_enabled", False):
        client_host = request.client.host if request.client is not None else ""
        try:
            if ip_address(client_host).is_loopback:
                return
        except ValueError:
            pass

    registry = _credential_registry(request)
    if not registry.validate(endpoint_id, request.cookies.get(ENDPOINT_COOKIE_NAME)):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "endpoint.authentication_required",
                "message": "A valid endpoint-scoped browser credential is required.",
            },
        )


@router.post("/{endpoint_id}/authenticate")
async def authenticate_endpoint(
    endpoint_id: str,
    request: Request,
    response: Response,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict[str, str | int]:
    """Exchange the private launch bearer for a bounded HttpOnly cookie.

    Examples:
        Establish browser authority without putting a token in request URLs:
        ```python
        POST /api/v1/agent-endpoints/studio-ui/authenticate
        Authorization: Bearer <launch credential>
        ```

    Args:
        endpoint_id: Exact endpoint identity selected by the Host manifest.
        request: FastAPI request carrying the Host credential registry.
        response: Response that receives the endpoint-scoped HttpOnly cookie.
        authorization: Launch bearer transferred to the browser in the URL fragment.

    Returns:
        dict[str, str | int]: Authenticated endpoint identity and remaining lifetime.

    Notes:
        The bearer is never accepted by execution routes and no query-token path exists.
    """

    registry = _credential_registry(request)
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not registry.validate(endpoint_id, token):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "endpoint.authentication_failed",
                "message": "The endpoint launch credential is invalid or expired.",
            },
        )
    ttl_seconds = registry.ttl_seconds(endpoint_id)
    response.set_cookie(
        key=ENDPOINT_COOKIE_NAME,
        value=token,
        max_age=ttl_seconds,
        httponly=True,
        secure=False,
        samesite="strict",
        path=f"/api/v1/agent-endpoints/{endpoint_id}",
    )
    return {"status": "authenticated", "endpoint_id": endpoint_id, "expires_in": ttl_seconds}


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


@router.get("/{endpoint_id}", response_model=EndpointDescriptor)
async def endpoint_descriptor(
    endpoint_id: str,
    request: Request,
    _credential: Annotated[None, Depends(require_endpoint_credential)],
) -> EndpointDescriptor:
    """Return the manifest-authoritative entry agent for one endpoint.

    The descriptor lets AG UI avoid presenting agent choices that the endpoint
    protocol cannot honor. Credential validation and manifest route resolution
    occur before any identity is returned.

    Examples:
        Inspect an AG UI endpoint:
            ```python
            GET /api/v1/agent-endpoints/studio-ui
            ```

        Read the entry agent:
            ```python
            agent_id = response["entry_agent_id"]
            ```

    Args:
        endpoint_id: Immutable endpoint identity selected by the Host manifest.
        request: FastAPI request carrying the installed Host.
        _credential: Validated endpoint credential dependency result.

    Returns:
        EndpointDescriptor: Exact endpoint and manifest entry-agent identities.

    Notes:
        The response contains no release metadata or credential material.
    """

    route = _route(_host(request), endpoint_id)
    return EndpointDescriptor(endpoint_id=endpoint_id, entry_agent_id=route.entry_agent_id)


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
    binding = await container.integration_ingress.session_store.get_binding(
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
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
    provisioning = await container.integration_ingress.provision_session(
        route_id=resolved.route_id,
        external_identity=external,
        request_scope=StorageScope(org_id=tenant_id, user_id=user_id),
        binding_id=f"binding-{digest[:32]}",
        ag_session_id=session_id,
        now=probe.received_at,
        title=body.title,
    )
    if provisioning.binding.ag_session_id != session_id:
        raise HTTPException(
            status_code=409,
            detail="Endpoint session binding conflicts with the canonical session identity.",
        )
    return EndpointSessionView(session_id=session_id, endpoint_id=endpoint_id)


async def _get_endpoint_session(container, *, route, identity, session_id: str) -> Session:
    external = _external_identity(identity=identity, session_id=session_id)
    binding = await _binding(container, route=route, external_identity=external)
    if binding.ag_session_id != session_id:
        raise HTTPException(status_code=404, detail="Endpoint session not found.")
    session = await container.session_store.get(session_id)
    tenant_id, user_id = _principal(identity)
    expected_source = (
        "ag_ui" if route.integration_kind is IntegrationKind.AG_UI else "agent_endpoint"
    )
    if (
        session is None
        or session.user_id != user_id
        or session.org_id != tenant_id
        or session.source != expected_source
        or session.external_ref != f"agent-endpoint:{route.endpoint_id}"
    ):
        raise HTTPException(status_code=404, detail="Endpoint session not found.")
    return session


@router.get("/{endpoint_id}/sessions", response_model=SessionListResponse)
async def list_endpoint_sessions(
    endpoint_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
    kind: SessionKind | None = Query(default=None),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=200),  # noqa: B008
    cursor: str | None = Query(default=None),  # noqa: B008
) -> SessionListResponse:
    """List sessions owned by one exact endpoint route.

    The scan applies identity, external-reference, and durable binding checks
    before returning a session. Its cursor advances over the underlying scoped
    session collection, so sessions belonging to other routes cannot poison or
    truncate the endpoint view.

    Examples:
        List the first page:
            ```python
            GET /api/v1/agent-endpoints/studio-ui/sessions?limit=20
            ```

        Continue from a cursor:
            ```python
            GET /api/v1/agent-endpoints/studio-ui/sessions?cursor=...
            ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.
        _credential: Validated endpoint credential dependency result.
        kind: Optional session-kind filter.
        limit: Maximum endpoint-owned sessions to return.
        cursor: Optional opaque cursor over the identity-scoped session list.

    Returns:
        SessionListResponse: Endpoint-owned sessions and an optional continuation cursor.

    Notes:
        Sessions with forged or stale `external_ref` values are excluded unless a
        matching durable endpoint binding also exists.
    """

    container = _host(request)
    route = _route(container, endpoint_id)
    tenant_id, user_id = _principal(identity)
    offset = decode_cursor(cursor)
    scan_offset = offset
    items: list[Session] = []
    batch_limit = max(50, min(500, limit * 2))
    source = "ag_ui" if route.integration_kind is IntegrationKind.AG_UI else "agent_endpoint"
    external_ref = f"agent-endpoint:{endpoint_id}"

    while len(items) < limit:
        batch = await container.session_store.list_for_user(
            user_id=user_id,
            org_id=tenant_id,
            kind=kind,
            limit=batch_limit,
            offset=scan_offset,
        )
        if not batch:
            return SessionListResponse(items=items, next_cursor=None)
        for session in batch:
            scan_offset += 1
            if session.source != source or session.external_ref != external_ref:
                continue
            try:
                bound = await _get_endpoint_session(
                    container,
                    route=route,
                    identity=identity,
                    session_id=session.session_id,
                )
            except HTTPException:
                continue
            items.append(bound)
            if len(items) == limit:
                return SessionListResponse(items=items, next_cursor=encode_cursor(scan_offset))
        if len(batch) < batch_limit:
            return SessionListResponse(items=items, next_cursor=None)

    return SessionListResponse(items=items, next_cursor=encode_cursor(scan_offset))


@router.get("/{endpoint_id}/sessions/{session_id}", response_model=Session)
async def get_endpoint_session(
    endpoint_id: str,
    session_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
) -> Session:
    """Return metadata for one endpoint-bound session.

    The lookup resolves the endpoint route and durable external binding before
    reading session metadata, preventing global-session access through this API.

    Examples:
        Read a session:
            ```python
            GET /api/v1/agent-endpoints/studio-ui/sessions/session-1
            ```

        Handle a cross-route session:
            ```python
            assert response.status_code == 404
            ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.
        _credential: Validated endpoint credential dependency result.

    Returns:
        Session: Metadata for the exact endpoint-bound session.

    Notes:
        Unknown and cross-route sessions use the same not-found response.
    """

    container = _host(request)
    route = _route(container, endpoint_id)
    return await _get_endpoint_session(
        container,
        route=route,
        identity=identity,
        session_id=session_id,
    )


@router.patch("/{endpoint_id}/sessions/{session_id}", response_model=Session)
async def update_endpoint_session(
    endpoint_id: str,
    session_id: str,
    body: EndpointSessionUpdate,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
) -> Session:
    """Update the title of one endpoint-bound session.

    Endpoint clients may change presentation metadata but cannot rewrite the
    immutable endpoint external reference used for route isolation.

    Examples:
        Rename a session:
            ```python
            PATCH /api/v1/agent-endpoints/studio-ui/sessions/session-1
            {"title": "New title"}
            ```

        Clear an automatically generated title:
            ```python
            PATCH /api/v1/agent-endpoints/studio-ui/sessions/session-1
            {"title": null}
            ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        body: Closed title update command.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.
        _credential: Validated endpoint credential dependency result.

    Returns:
        Session: Updated endpoint-bound session metadata.

    Notes:
        Endpoint external references cannot be changed through this route.
    """

    container = _host(request)
    route = _route(container, endpoint_id)
    session = await _get_endpoint_session(
        container,
        route=route,
        identity=identity,
        session_id=session_id,
    )
    updated = await container.session_store.update(
        session.session_id,
        title=body.title,
        title_source="manual",
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Endpoint session not found.")
    return updated


@router.delete("/{endpoint_id}/sessions/{session_id}", status_code=204)
async def delete_endpoint_session(
    endpoint_id: str,
    session_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
) -> Response:
    """Delete one endpoint-bound session.

    Durable route binding verification happens before deletion, so a credential
    for one endpoint cannot delete another endpoint's session.

    Examples:
        Delete an endpoint session:
            ```python
            DELETE /api/v1/agent-endpoints/studio-ui/sessions/session-1
            ```

        Reject a cross-route deletion:
            ```python
            assert response.status_code == 404
            ```

    Args:
        endpoint_id: Immutable manifest endpoint identity.
        session_id: Public endpoint session identity.
        request: FastAPI request carrying the installed Host.
        identity: Authenticated endpoint principal.
        _credential: Validated endpoint credential dependency result.

    Returns:
        Response: Empty HTTP 204 response after successful deletion.

    Notes:
        Deleting a session does not mutate the immutable Host manifest.
    """

    container = _host(request)
    route = _route(container, endpoint_id)
    session = await _get_endpoint_session(
        container,
        route=route,
        identity=identity,
        session_id=session_id,
    )
    await container.session_store.delete(session.session_id)
    return Response(status_code=204)


async def _parse_ingress(
    request: Request,
) -> tuple[EndpointIngressBody, tuple[StarletteUploadFile, ...]]:
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
    uploads: list[StarletteUploadFile] = []
    for field_name, value in form.multi_items():
        if not isinstance(value, StarletteUploadFile):
            continue
        if field_name != "files":
            raise ValueError(f"Unexpected endpoint upload field: {field_name!r}.")
        uploads.append(value)
    return body, tuple(uploads)


def _resource_policy(container) -> ResourceIngressPolicy:
    resource_ingress = getattr(container.integration_ingress, "resource_ingress", None)
    policy = getattr(resource_ingress, "policy", None)
    if not isinstance(policy, ResourceIngressPolicy):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "endpoint.attachment_policy_unavailable",
                "message": "Endpoint attachment policy is unavailable.",
            },
        )
    return policy


def _raise_attachment_http_error(error: ResourceIngressError) -> Never:
    status_code = 415 if error.code == "integration.attachment_type_rejected" else 413
    raise HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": str(error)},
    ) from error


@router.post("/{endpoint_id}/ingress")
async def endpoint_ingress(
    endpoint_id: str,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
        raise HTTPException(
            status_code=400,
            detail={
                "code": "endpoint.ingress_body_invalid",
                "message": "Invalid endpoint ingress body.",
            },
        ) from exc
    external = _external_identity(identity=identity, session_id=body.session_id)
    await _binding(container, route=route, external_identity=external)
    declared = [
        IngressAttachment(
            attachment_id=item.artifact_id,
            source_kind="artifact",
            source_id=item.artifact_id,
            filename=item.filename or item.artifact_id,
            content_type=item.content_type or "application/octet-stream",
            size_bytes=item.size_bytes,
        )
        for item in body.attachments
    ]
    policy = _resource_policy(container)
    if len(declared) + len(uploads) > policy.max_count:
        _raise_attachment_http_error(
            ResourceIngressError(
                code="integration.attachment_count_exceeded",
                message=f"Ingress contains more than {policy.max_count} attachments.",
            )
        )
    verified_files: list[VerifiedAttachment] = []
    uploaded_total = 0
    for index, upload in enumerate(uploads):
        digest = sha256(f"{body.idempotency_key}\0{index}".encode()).hexdigest()[:24]
        attachment_id = f"upload-{digest}"
        remaining = policy.max_total_bytes - uploaded_total
        read_limit = min(policy.max_file_bytes, max(0, remaining))
        overflow_code: Literal[
            "integration.attachment_too_large",
            "integration.attachment_total_exceeded",
        ] = (
            "integration.attachment_total_exceeded"
            if read_limit < policy.max_file_bytes
            else "integration.attachment_too_large"
        )
        try:
            data = await read_bounded_attachment_bytes(
                upload.read,
                max_bytes=read_limit,
                attachment_id=attachment_id,
                overflow_code=overflow_code,
            )
        except ResourceIngressError as exc:
            _raise_attachment_http_error(exc)
        uploaded_total += len(data)
        declared.append(
            IngressAttachment(
                attachment_id=attachment_id,
                source_kind="provider_file",
                source_id=attachment_id,
                filename=upload.filename or attachment_id,
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
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
        The stream polls only the canonical semantic store; no broadcast side channel exists.
    """
    container, _, binding = await _session_binding(request, endpoint_id, session_id, identity)
    cursor_start = _stream_cursor(
        after_cursor=after_cursor,
        last_event_id=request.headers.get("last-event-id"),
    )

    async def generate():
        cursor = cursor_start
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


def _stream_cursor(*, after_cursor: int | None, last_event_id: str | None) -> int | None:
    """Resolve the exclusive semantic cursor for one SSE connection.

    The explicit query cursor establishes the initial subscription position.
    On browser reconnect, a valid WHATWG `Last-Event-ID` can only advance that
    position, so already delivered durable Events are not replayed.

    Examples:
        Resolve a fresh stream:
        ```python
        assert _stream_cursor(after_cursor=None, last_event_id=None) is None
        ```

        Advance a reconnecting stream:
        ```python
        assert _stream_cursor(after_cursor=4, last_event_id="9") == 9
        ```

    Args:
        after_cursor: Optional exclusive cursor supplied in the request query.
        last_event_id: Optional SSE Event ID supplied by the user agent.

    Returns:
        int | None: The greatest valid supplied cursor, or `None` for initial history.

    Notes:
        Malformed or negative `Last-Event-ID` values fail closed with HTTP 400.
    """
    if last_event_id is None or not last_event_id.strip():
        return after_cursor
    raw = last_event_id.strip()
    if not raw.isdecimal():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "endpoint.last_event_id_invalid",
                "message": "Last-Event-ID must be a non-negative integer cursor.",
            },
        )
    reconnect_cursor = int(raw)
    if after_cursor is None:
        return reconnect_cursor
    return max(after_cursor, reconnect_cursor)


@router.post("/{endpoint_id}/sessions/{session_id}/cancel")
async def cancel_endpoint_turn(
    endpoint_id: str,
    session_id: str,
    body: EndpointCancelBody,
    request: Request,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
    _credential: Annotated[None, Depends(require_endpoint_credential)],
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
    artifacts = container.artifact_factory.for_public_execution(
        StorageScope(org_id=identity.org_id, user_id=identity.user_id)
    )
    artifact = await artifacts.get_by_id(artifact_id)
    labels = artifact.labels if artifact is not None else {}
    if (
        artifact is None
        or not artifact_belongs_to_identity(identity, artifact)
        or labels.get("route_id") != route.route_id
    ):
        raise HTTPException(status_code=404, detail="Artifact not found.")
    data = await artifacts.load_bytes_by_id(artifact_id)
    return Response(
        content=data,
        media_type=artifact.mime or "application/octet-stream",
        headers={"X-AetherGraph-Artifact-Id": artifact.artifact_id},
    )
