# /artifacts

import mimetypes
import os
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from fastapi.responses import RedirectResponse

from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.contracts.storage.artifact_index import Artifact
from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas import (
    ArtifactListResponse,
    ArtifactMeta,
    ArtifactSearchHit,
    ArtifactSearchRequest,
    ArtifactSearchResponse,
)

router = APIRouter(tags=["artifacts"])


# -------- Helpers  -------- #


def _tenant_label_filters(identity: RequestIdentity) -> dict[str, str]:
    """
    Convert RequestIdentity into artifact label filters.

    All modes (cloud/demo/local) get org_id + user_id set, so we just use that.
    """
    org_id, user_id = identity.tenant_key
    filters: dict[str, str] = {}

    if org_id is not None:
        filters["org_id"] = org_id
    if user_id is not None:
        filters["user_id"] = user_id

    return filters


def _extract_tags(labels: dict[str, Any]) -> list[str]:
    """
    Conventions:
    - labels["tags"] may be a list[str] or comma-separated str
    """
    tags = labels.get("tags")
    if isinstance(tags, list):
        return [str(t) for t in tags]
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _extract_scope_id(a: Artifact) -> str | None:
    """
    Conventions:
    - labels["scope_id"] is preferred
    - labels["scope"] is legacy
    - fallback to run_id if no scope label found
    """
    labels = a.labels or {}
    scope = labels.get("scope_id") or labels.get("scope")  # legacy
    if scope is not None:
        return str(scope)
    return a.run_id  # fallback to run_id if no scope label found


def _guess_mime(a: Artifact) -> str:
    # 1) explicit mime wins
    if a.mime:
        return a.mime

    # 2) infer from URI / filename
    mime = None
    if a.uri:
        guessed, _ = mimetypes.guess_type(a.uri)
        if guessed:
            mime = guessed

    # 3) heuristics from kind (optional but nice)
    if not mime and a.kind:
        k = a.kind.lower()
        if any(x in k for x in ["log", "text", "stdout", "stderr"]):
            mime = "text/plain"
        elif "json" in k:
            mime = "application/json"
        elif "csv" in k:
            mime = "text/csv"
        elif "markdown" in k or "md" in k:
            mime = "text/markdown"

    # 4) fallback
    return mime or "application/octet-stream"


def _artifact_to_meta(a: Artifact) -> ArtifactMeta:
    """
    Convert Artifact to ArtifactMeta schema.
    """
    labels = a.labels or {}

    out = ArtifactMeta(
        artifact_id=a.artifact_id,
        kind=a.kind,
        mime_type=_guess_mime(a),
        size=a.bytes,
        scope_id=_extract_scope_id(a) or "unknown_scope",
        tags=_extract_tags(labels),
        created_at=a.created_at,  # pydantic will parse ISO str -> datetime
        uri=a.uri,
        pinned=a.pinned,
        preview_uri=a.preview_uri,
        run_id=a.run_id,
        graph_id=a.graph_id,
        node_id=a.node_id if getattr(a, "node_id", None) else None,
        session_id=a.session_id if getattr(a, "session_id", None) else None,
        filename=labels.get("filename"),
    )
    return out


# -------- API Endpoints -------- #
@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    scope_id: Annotated[str | None, Query()] = None,
    run_id: Annotated[str | None, Query()] = None,
    session_id: Annotated[str | None, Query()] = None,
    kind: Annotated[str | None, Query()] = None,
    tags: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    container = current_services()
    index = getattr(container, "artifact_index", None)
    if index is None:
        return ArtifactListResponse(artifacts=[], next_cursor=None)

    offset = decode_cursor(cursor.strip() if cursor else None)

    # label_filters: dict[str, Any] = {}

    # if scope_id and scope_id.strip():
    #     label_filters["scope_id"] = scope_id.strip()

    # if tags and tags.strip():
    #     tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    #     if tag_list:
    #         label_filters["tags"] = tag_list

    # # 🔹 Tenant scoping: org_id + user_id
    # label_filters.update(_tenant_label_filters(identity))

    label_filters: dict[str, Any] = {}

    # execution scopes
    if run_id and run_id.strip():
        label_filters["run_id"] = run_id.strip()
    if session_id and session_id.strip():
        label_filters["session_id"] = session_id.strip()

    # memory scope (keep for “overview” / RAG-style scoping)
    if scope_id and scope_id.strip():
        label_filters["scope_id"] = scope_id.strip()

    if tags and tags.strip():
        label_filters["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    print("Label filters:", label_filters)
    label_filters.update(_tenant_label_filters(identity))

    artifacts = await index.search(
        kind=kind.strip() if kind and kind.strip() else None,
        labels=label_filters or None,
        metric=None,
        mode=None,
        limit=limit,
        offset=offset,
    )
    metas = [_artifact_to_meta(a) for a in artifacts]
    next_cursor = encode_cursor(offset + limit) if len(artifacts) == limit else None
    return ArtifactListResponse(artifacts=metas, next_cursor=next_cursor)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactMeta)
async def get_artifact(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactMeta:
    """
    Get single artifact metadata.
    """
    container = current_services()
    index = getattr(container, "artifact_index", None)
    rm = getattr(container, "run_manager", None)
    if index is None or (identity.mode == "demo" and rm is None):
        raise HTTPException(status_code=503, detail="Artifact index not configured")

    artifact = await index.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    meta = _artifact_to_meta(artifact)
    return meta


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Response:
    container = current_services()
    index = getattr(container, "artifact_index", None)
    store = getattr(container, "artifacts", None)
    rm = getattr(container, "run_manager", None)
    if index is None or store is None or (identity.client_id and rm is None):
        raise HTTPException(status_code=503, detail="Artifact services not configured")

    artifact = await index.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    # If user provided a fully qualified preview URI (e.g. S3 signed URL)
    if artifact.preview_uri and str(artifact.preview_uri).startswith(("http://", "https://")):
        return RedirectResponse(artifact.preview_uri)

    # Otherwise, stream raw bytes from the artifact store.
    data = await store.load_artifact_bytes(artifact.uri)

    # Derive a filename that's at least somewhat meaningful
    labels = artifact.labels or {}
    filename = (
        labels.get("filename")
        or (os.path.basename(artifact.uri) if artifact.uri else None)
        or artifact.artifact_id
    )

    media_type = artifact.mime or "application/octet-stream"

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Length": str(len(data)),
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-AetherGraph-Artifact-Id": artifact.artifact_id,
        },
    )


@router.post("/artifacts/{artifact_id}/pin")
async def pin_artifact(
    artifact_id: str,
    pinned: Annotated[bool, Body()] = True,
    identity: Annotated[RequestIdentity, Depends(get_identity)] = None,
) -> dict:
    """
    Mark/unmark an artifact as pinned in the index.

    Pinned artifacts can be treated as "keep" in GC policies or highlighted in UIs.
    """
    container = current_services()
    rm = getattr(container, "run_manager", None)
    index = getattr(container, "artifact_index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Artifact index not configured")

    if identity.client_id and rm is None:
        # Can't enforce client scoping without RunManager
        raise HTTPException(status_code=503, detail="Run manager not configured")

    artifact = await index.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    await index.pin(artifact_id, pinned=pinned)
    return {"artifact_id": artifact_id, "pinned": pinned}


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactListResponse)
async def list_run_artifacts(
    run_id: str,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    container = current_services()
    index = getattr(container, "artifact_index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Artifact index not configured")

    offset = decode_cursor(cursor.strip() if cursor else None)

    label_filters: dict[str, Any] = {"run_id": run_id}
    label_filters.update(_tenant_label_filters(identity))

    artifacts = await index.search(
        labels=label_filters,
        limit=limit,
        offset=offset,
    )

    metas = [_artifact_to_meta(a) for a in artifacts]
    next_cursor = encode_cursor(offset + limit) if len(artifacts) == limit else None
    return ArtifactListResponse(artifacts=metas, next_cursor=next_cursor)


@router.get("/sessions/{session_id}/artifacts", response_model=ArtifactListResponse)
async def list_session_artifacts(
    session_id: str,
    cursor: Annotated[str | None, Query()] = None,  # noqa: B008
    limit: Annotated[int, Query(ge=1, le=200)] = 50,  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    container = current_services()
    index = getattr(container, "artifact_index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Artifact index not configured")

    offset = decode_cursor(cursor.strip() if cursor else None)

    label_filters: dict[str, Any] = {"session_id": session_id}
    label_filters.update(_tenant_label_filters(identity))

    artifacts = await index.search(
        labels=label_filters,
        limit=limit,
        offset=offset,
    )

    metas = [_artifact_to_meta(a) for a in artifacts]
    next_cursor = encode_cursor(offset + limit) if len(artifacts) == limit else None
    return ArtifactListResponse(artifacts=metas, next_cursor=next_cursor)


@router.post("/artifacts/search", response_model=ArtifactSearchResponse)
async def search_artifacts(
    req: ArtifactSearchRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> ArtifactSearchResponse:
    """
    Structured search over artifacts via the artifact index.

    We interpret fields on ArtifactSearchRequest in a flexible way:
      - kind: optional artifact kind filter
      - scope_id: maps to labels["scope_id"]
      - tags: optional list[str] or comma-separated string -> labels["tags"]
      - labels: optional extra label filters
      - metric + mode: if provided, used for ranking (and required for best-only)
      - limit: max results
      - best_only: if True, use index.best(...) and return a single hit

    Tenant scoping is enforced via org_id/user_id/client_id/app_id from RequestIdentity.
    """
    container = current_services()
    index = getattr(container, "artifact_index", None)
    if index is None:
        return ArtifactSearchResponse(results=[])

    kind = getattr(req, "kind", None)
    scope_id = getattr(req, "scope_id", None)
    tags = getattr(req, "tags", None)
    extra_labels = getattr(req, "labels", None)
    metric = getattr(req, "metric", None)
    mode = getattr(req, "mode", None)
    limit = getattr(req, "limit", 50)
    best_only = getattr(req, "best_only", False)

    label_filter: dict[str, Any] = {}

    if scope_id:
        label_filter["scope_id"] = scope_id

    # Handle tags, may be list or comma-separated str
    if tags:
        if isinstance(tags, str):
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        elif isinstance(tags, list):
            tag_list = [str(t) for t in tags]
        else:
            tag_list = []
        if tag_list:
            label_filter["tags"] = tag_list

    if extra_labels:
        label_filter.update(extra_labels)

    # 🔹 Tenant scoping
    tenant_filters = _tenant_label_filters(identity)
    label_filter.update(tenant_filters)

    hits: list[ArtifactSearchHit] = []

    if best_only and metric and mode:
        best = await index.best(
            kind=kind or "",
            metric=metric,
            mode=mode,
            filters=label_filter or None,
        )
        if best is not None:
            score = float(best.metrics.get(metric, 0.0)) if best.metrics else 0.0
            hits.append(
                ArtifactSearchHit(
                    artifact=_artifact_to_meta(best),
                    score=score,
                )
            )
        return ArtifactSearchResponse(results=hits)

    artifacts = await index.search(
        kind=kind,
        labels=label_filter or None,
        metric=metric,
        mode=mode,
        limit=limit,
    )

    for a in artifacts:
        score = 1.0
        if metric and a.metrics:
            score = float(a.metrics.get(metric, 0.0))
        hits.append(
            ArtifactSearchHit(
                artifact=_artifact_to_meta(a),
                score=score,
            )
        )

    return ArtifactSearchResponse(results=hits)
