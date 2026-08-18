# /artifacts

import mimetypes
from time import perf_counter
from typing import Annotated, Any
import unicodedata

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response  # type: ignore
from fastapi.responses import RedirectResponse  # type: ignore

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
from aethergraph.storage.contracts import (
    ArtifactMetricOrder,
    PageRequest,
    SearchMode,
    StorageScope,
)

from .deps import RequestIdentity, artifact_belongs_to_identity, get_identity
from .schemas.artifacts import (
    ArtifactListResponse,
    ArtifactMeta,
    ArtifactSearchHit,
    ArtifactSearchRequest,
    ArtifactSearchResponse,
)


def _latin1_safe(s: str, fallback: str = "") -> str:
    try:
        s.encode("latin-1")
        return s
    except UnicodeEncodeError:
        # Fallback: strip accents & non-ascii
        ascii_guess = (
            unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii") or fallback
        )
        return ascii_guess or "artifact"


router = APIRouter(tags=["artifacts"])

_DEPRECATED_SEARCH_IDENTITY_LABELS = frozenset({"app_id", "application_id", "client_id"})
_PROMOTED_SEARCH_FIELDS = frozenset({"kind", "scope_id", "tags"})


# -------- Helpers  -------- #


def _artifact_service(
    identity: RequestIdentity,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
) -> CanonicalPublicArtifactFacade:
    container = current_services()
    org_id, user_id = identity.tenant_key
    return container.artifact_factory.for_public_execution(
        StorageScope(
            org_id=org_id,
            user_id=user_id,
            run_id=run_id,
            session_id=session_id,
        )
    )


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
        occurrence_id=getattr(a, "occurrence_id", None),
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


async def _search_canonical_artifacts(
    req: ArtifactSearchRequest,
    facade: CanonicalPublicArtifactFacade,
) -> ArtifactSearchResponse:
    """Map the frozen Artifact search request onto exact canonical query paths."""
    query = req.query.strip() if req.query and req.query.strip() else None
    kind = _required_optional_text("kind", req.kind)
    scope_id = _required_optional_text("scope_id", req.scope_id)
    metric = _required_optional_text("metric", req.metric)
    tags = _canonical_search_tags(req.tags)
    labels = dict(req.labels)
    _validate_canonical_search_labels(labels)
    if scope_id is not None:
        labels["scope_id"] = scope_id
    if isinstance(req.limit, bool) or not 1 <= req.limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")

    has_metric_order = req.mode is not None
    if query is not None:
        if metric is not None or has_metric_order or req.best_only:
            raise HTTPException(
                status_code=422,
                detail="text search cannot be combined with metric ranking or best_only",
            )
        metadata = dict(labels)
        if kind is not None:
            metadata["kind"] = kind
        results = await facade.search_public_artifacts(
            query=query,
            mode=SearchMode.LEXICAL,
            top_k=req.limit,
            tags=tags,
            metadata=metadata,
        )
        return ArtifactSearchResponse(
            hits=[
                ArtifactSearchHit(artifact=_artifact_to_meta(result.artifact), score=result.score)
                for result in results
            ]
        )

    if (metric is None) != (req.mode is None):
        raise HTTPException(
            status_code=422,
            detail="metric and mode must be supplied together",
        )
    if req.best_only and metric is None:
        raise HTTPException(
            status_code=422,
            detail="best_only requires metric and mode",
        )
    metric_order = ArtifactMetricOrder(req.mode) if req.mode is not None else None
    page = await facade.query_public_artifacts(
        PageRequest(limit=1 if req.best_only else req.limit),
        kind=kind,
        tags=tags,
        labels=labels,
        metric=metric,
        metric_order=metric_order,
    )
    return ArtifactSearchResponse(
        hits=[
            ArtifactSearchHit(
                artifact=_artifact_to_meta(artifact),
                score=float(artifact.metrics[metric]) if metric is not None else 1.0,
            )
            for artifact in page.items
        ]
    )


def _required_optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{name} must be non-empty when supplied")
    return normalized


def _canonical_search_tags(values: list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise HTTPException(status_code=422, detail="tags must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise HTTPException(status_code=422, detail="tags must not contain duplicates")
    return tuple(sorted(normalized))


def _validate_canonical_search_labels(labels: dict[str, Any]) -> None:
    forbidden = sorted(
        (_DEPRECATED_SEARCH_IDENTITY_LABELS | _PROMOTED_SEARCH_FIELDS).intersection(labels)
    )
    if forbidden:
        names = ", ".join(forbidden)
        raise HTTPException(
            status_code=422,
            detail=f"labels contains reserved or deprecated search fields: {names}",
        )


# -------- API Endpoints -------- #
@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    scope_id: Annotated[str | None, Query()] = None,
    run_id: Annotated[str | None, Query()] = None,
    session_id: Annotated[str | None, Query()] = None,
    kind: Annotated[str | None, Query()] = None,
    tags: Annotated[str | None, Query()] = None,
    pinned: Annotated[bool | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    response: Response = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    # print(f"list_artifacts called with scope_id={scope_id}, run_id={run_id}, session_id={session_id}, kind={kind}, tags={tags}, cursor={cursor}, limit={limit}, identity={identity}")
    facade = _artifact_service(
        identity,
        run_id=run_id.strip() if run_id and run_id.strip() else None,
        session_id=session_id.strip() if session_id and session_id.strip() else None,
    )
    label_filters: dict[str, Any] = {}

    if scope_id and scope_id.strip():
        label_filters["scope_id"] = scope_id.strip()

    required_tags = tuple(t.strip() for t in tags.split(",") if t.strip()) if tags else ()

    started_at = perf_counter()
    page = await facade.query_public_artifacts(
        PageRequest(limit=limit, cursor=cursor.strip() if cursor else None),
        kind=kind.strip() if kind and kind.strip() else None,
        tags=required_tags,
        labels=label_filters,
        pinned=pinned,
    )
    metas = [_artifact_to_meta(a) for a in page.items]
    if response is not None:
        response.headers["X-AetherGraph-Artifact-Query-Ms"] = (
            f"{(perf_counter() - started_at) * 1000:.2f}"
        )
    return ArtifactListResponse(artifacts=metas, next_cursor=page.next_cursor)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactMeta)
async def get_artifact(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactMeta:
    """
    Get single artifact metadata.
    """
    artifact = await _artifact_service(identity).get_by_id(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    if not artifact_belongs_to_identity(identity, artifact):
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    meta = _artifact_to_meta(artifact)
    return meta


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Response:
    facade = _artifact_service(identity)
    artifact = await facade.get_by_id(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    if not artifact_belongs_to_identity(identity, artifact):
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    # If user provided a fully qualified preview URI (e.g. S3 signed URL)
    if artifact.preview_uri and str(artifact.preview_uri).startswith(("http://", "https://")):
        return RedirectResponse(artifact.preview_uri)

    # Otherwise, stream raw bytes from the artifact store.
    data = await facade.load_bytes_by_id(artifact_id)

    # Derive a filename that's at least somewhat meaningful
    labels = artifact.labels or {}
    filename = labels.get("filename") or artifact.artifact_id

    media_type = artifact.mime or "application/octet-stream"

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Length": str(len(data)),
            "Content-Disposition": f'attachment; filename="{_latin1_safe(filename)}"',
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
    facade = _artifact_service(identity)
    artifact = await facade.get_by_id(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
    if not artifact_belongs_to_identity(identity, artifact):
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")

    await facade.pin(artifact_id, pinned=pinned)
    return {"artifact_id": artifact_id, "pinned": pinned}


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactListResponse)
async def list_run_artifacts(
    run_id: str,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    response: Response = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    started_at = perf_counter()
    page = await _artifact_service(identity, run_id=run_id).query_public_artifacts(
        PageRequest(limit=limit, cursor=cursor.strip() if cursor else None)
    )
    metas = [_artifact_to_meta(a) for a in page.items]
    if response is not None:
        response.headers["X-AetherGraph-Artifact-Query-Ms"] = (
            f"{(perf_counter() - started_at) * 1000:.2f}"
        )
    return ArtifactListResponse(artifacts=metas, next_cursor=page.next_cursor)


@router.get("/sessions/{session_id}/artifacts", response_model=ArtifactListResponse)
async def list_session_artifacts(
    session_id: str,
    cursor: Annotated[str | None, Query()] = None,  # noqa: B008
    limit: Annotated[int, Query(ge=1, le=200)] = 50,  # noqa: B008
    response: Response = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    started_at = perf_counter()
    page = await _artifact_service(identity, session_id=session_id).query_public_artifacts(
        PageRequest(limit=limit, cursor=cursor.strip() if cursor else None)
    )
    metas = [_artifact_to_meta(a) for a in page.items]
    if response is not None:
        response.headers["X-AetherGraph-Artifact-Query-Ms"] = (
            f"{(perf_counter() - started_at) * 1000:.2f}"
        )
    return ArtifactListResponse(artifacts=metas, next_cursor=page.next_cursor)


@router.post("/artifacts/search", response_model=ArtifactSearchResponse)
async def search_artifacts(
    req: ArtifactSearchRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> ArtifactSearchResponse:
    """
    Structured search over the selected canonical Artifact provider.
    """
    return await _search_canonical_artifacts(req, _artifact_service(identity))
