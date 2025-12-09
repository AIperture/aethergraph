from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aethergraph.api.v1.deps import RequestIdentity, get_identity

router = APIRouter()


class IdentityResponse(BaseModel):
    mode: str
    user_id: str | None
    org_id: str | None
    roles: list[str]
    client_id: str | None


@router.get("/whoami", response_model=IdentityResponse)
def whoami(identity: RequestIdentity = Depends(get_identity)):  # noqa: B008
    return IdentityResponse(
        mode=identity.mode,
        user_id=identity.user_id,
        org_id=identity.org_id,
        roles=identity.roles,
        client_id=identity.client_id,
    )
