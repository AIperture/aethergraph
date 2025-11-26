# stub for auth/identity management dependencies


from fastapi import Header, Request
from pydantic import BaseModel


class RequestIdentity(BaseModel):
    user_id: str | None = None
    org_id: str | None = None
    roles: list[str] = []


async def get_identity(
    request: Request,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    x_org_id: str | None = Header(None, alias="X-Org-ID"),
    x_roles: str | None = Header(None, alias="X-Roles"),
) -> RequestIdentity:
    """
    Identity extraction hook.

    - Dev/local: if no headers, fall back to a default "local" identity.
    - Production: run AG behind an auth gateway that injects these headers
      OR validate a JWT and populate identity here.
    """

    # TODO: parse request.headers.get("Authorization") for a JWT token,

    roles = x_roles.split(",") if x_roles else []

    if not x_user_id and not x_org_id:
        # Dev / local fallback
        return RequestIdentity(user_id="local", org_id="local", roles=["dev"])

    return RequestIdentity(user_id=x_user_id, org_id=x_org_id, roles=roles)
