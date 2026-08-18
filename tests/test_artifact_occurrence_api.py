from __future__ import annotations

from fastapi import HTTPException
import pytest

from aethergraph.api.v1 import artifacts as artifacts_api
from aethergraph.api.v1.schemas.artifacts import ArtifactSearchRequest
from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.artifacts import PublicArtifactSearchHit
from aethergraph.storage.contracts import ArtifactMetricOrder, Page, PageRequest, SearchMode


class _CanonicalSearchFacade:
    def __init__(self) -> None:
        self.text_call: dict[str, object] | None = None
        self.structural_call: dict[str, object] | None = None

    async def search_public_artifacts(self, **kwargs) -> tuple[PublicArtifactSearchHit, ...]:
        self.text_call = kwargs
        return (
            PublicArtifactSearchHit(
                artifact=_canonical_public_artifact(),
                score=0.75,
                mode=SearchMode.LEXICAL,
            ),
        )

    async def query_public_artifacts(self, page, **kwargs) -> Page[Artifact]:
        self.structural_call = {"page": page, **kwargs}
        return Page(items=(_canonical_public_artifact(),), next_cursor=None)


def _canonical_public_artifact() -> Artifact:
    return Artifact(
        artifact_id="artifact-1",
        kind="report",
        mime="text/plain",
        bytes=10,
        created_at="2026-08-15T00:00:00+00:00",
        labels={"scope_id": "logical-scope", "tags": ["final", "reviewed"]},
        metrics={"quality": 0.9},
    )


@pytest.mark.asyncio
async def test_canonical_artifact_search_mapper_uses_exact_lexical_mode() -> None:
    facade = _CanonicalSearchFacade()

    response = await artifacts_api._search_canonical_artifacts(
        ArtifactSearchRequest(
            query="  migration report  ",
            scope_id=" logical-scope ",
            kind=" report ",
            tags=[" reviewed ", "final"],
            labels={"stage": "published"},
            limit=7,
        ),
        facade,  # type: ignore[arg-type]
    )

    assert facade.text_call == {
        "query": "migration report",
        "mode": SearchMode.LEXICAL,
        "top_k": 7,
        "tags": ("final", "reviewed"),
        "metadata": {
            "kind": "report",
            "scope_id": "logical-scope",
            "stage": "published",
        },
    }
    assert facade.structural_call is None
    assert response.hits[0].score == 0.75
    assert response.hits[0].artifact is not None
    assert response.hits[0].artifact.artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_canonical_artifact_search_mapper_uses_indexed_metric_ranking() -> None:
    facade = _CanonicalSearchFacade()

    response = await artifacts_api._search_canonical_artifacts(
        ArtifactSearchRequest(
            kind="report",
            tags=["final"],
            labels={"stage": "published"},
            metric="quality",
            mode="max",
            limit=20,
            best_only=True,
        ),
        facade,  # type: ignore[arg-type]
    )

    assert facade.text_call is None
    assert facade.structural_call == {
        "page": PageRequest(limit=1),
        "kind": "report",
        "tags": ("final",),
        "labels": {"stage": "published"},
        "metric": "quality",
        "metric_order": ArtifactMetricOrder.MAXIMUM,
    }
    assert response.hits[0].score == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"query": "report", "metric": "quality", "mode": "max"}, "text search cannot be combined"),
        ({"metric": "quality"}, "metric and mode must be supplied together"),
        ({"best_only": True}, "best_only requires metric and mode"),
        ({"tags": ["final", " final "]}, "tags must not contain duplicates"),
        ({"labels": {"app_id": "legacy"}}, "app_id"),
        ({"labels": {"tags": ["final"]}}, "tags"),
    ],
)
async def test_canonical_artifact_search_mapper_rejects_ambiguous_requests(
    payload: dict[str, object],
    detail: str,
) -> None:
    facade = _CanonicalSearchFacade()
    with pytest.raises(HTTPException) as failure:
        await artifacts_api._search_canonical_artifacts(
            ArtifactSearchRequest(**payload),
            facade,  # type: ignore[arg-type]
        )

    assert getattr(failure.value, "status_code", None) == 422
    assert detail in str(getattr(failure.value, "detail", ""))
    assert facade.text_call is None
    assert facade.structural_call is None
