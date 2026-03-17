from fastapi import APIRouter, Depends  # type: ignore
from pydantic import BaseModel  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity, get_identity

router = APIRouter()


class IdentityResponse(BaseModel):
    mode: str
    user_id: str | None
    org_id: str | None
    roles: list[str]
    client_id: str | None
    grant_id: str | None = None
    auth_source: str | None = None
    catalog_scope: dict[str, list[str]] | None = None


@router.get("/whoami", response_model=IdentityResponse)
def whoami(identity: RequestIdentity = Depends(get_identity)):  # noqa: B008
    return IdentityResponse(
        mode=identity.mode,
        user_id=identity.user_id,
        org_id=identity.org_id,
        roles=identity.roles,
        client_id=identity.client_id,
        grant_id=identity.grant_id,
        auth_source=identity.auth_source,
        catalog_scope=identity.catalog_scope,
    )
