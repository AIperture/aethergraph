# /artifacts

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, Response

from .deps import RequestIdentity, get_identity
from .schemas import (
    ArtifactListResponse,
    ArtifactMeta,
    ArtifactSearchHit,
    ArtifactSearchRequest,
    ArtifactSearchResponse,
)

router = APIRouter(tags=["artifacts"])


@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    scope_id: str = Query(None),  # noqa: B008
    kind: str | None = Query(None),  # noqa: B008
    tags: str | None = Query(None, description="Comma-separated list of tags to filter"),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactListResponse:
    """
    List artifacts (metadata only).

    TODO:
      - Integrate with ArtifactStore + index.
    """
    now = datetime.utcnow()
    dummy = ArtifactMeta(
        artifact_id="art-1",
        kind="file",
        mime_type="text/plain",
        size=123,
        scope_id=scope_id or "stub_scope",
        tags=["stub"],
        created_at=now - timedelta(minutes=10),
        uri="fs://stub/path.txt",
    )
    return ArtifactListResponse(artifacts=[dummy], next_cursor=None)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactMeta)
async def get_artifact(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactMeta:
    """
    Get single artifact metadata.

    TODO:
      - Load metadata from ArtifactStore.
    """
    now = datetime.utcnow()
    return ArtifactMeta(
        artifact_id=artifact_id,
        kind="file",
        mime_type="text/plain",
        size=123,
        scope_id="stub_scope",
        tags=["stub"],
        created_at=now,
        uri="fs://stub/path.txt",
    )


@router.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Response:
    """
    Stream artifact content.

    TODO:
      - Stream from ArtifactStore (FS/S3/etc.).
      - Set proper Content-Type and headers.
    """
    content = f"Stub content for artifact {artifact_id}\n"
    return Response(content=content, media_type="text/plain")


@router.post("/artifacts/search", response_model=ArtifactSearchResponse)
async def search_artifacts(
    req: ArtifactSearchRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactSearchResponse:
    """
    Semantic/keyword search over artifacts.

    TODO:
      - Plug into your artifact index (e.g. embeddings).
    """
    now = datetime.utcnow()
    dummy_meta = ArtifactMeta(
        artifact_id="art-hit",
        kind="file",
        mime_type="text/plain",
        size=456,
        scope_id=req.scope_id or "stub_scope",
        tags=["search_stub"],
        created_at=now,
        uri="fs://stub/path_hit.txt",
    )
    hit = ArtifactSearchHit(score=0.97, artifact=dummy_meta)
    return ArtifactSearchResponse(hits=[hit])
