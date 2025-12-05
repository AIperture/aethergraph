# stub for auth/identity management dependencies


from typing import Literal

from fastapi import Header, Request
from pydantic import BaseModel, Field


class RequestIdentity(BaseModel):
    user_id: str | None = None
    org_id: str | None = None
    roles: list[str] = Field(default_factory=list)

    # Demo-only/browser identity
    client_id: str | None = None

    # How this request is “authenticated”
    mode: Literal["cloud", "demo", "local"] = "local"


async def get_identity(
    request: Request,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    x_org_id: str | None = Header(None, alias="X-Org-ID"),
    x_roles: str | None = Header(None, alias="X-Roles"),
    x_client_id: str | None = Header(None, alias="X-Client-ID"),
) -> RequestIdentity:
    """
    Identity extraction hook.

    Modes:
    - CLOUD: auth gateway injects X-User-ID / X-Org-ID (optionally X-Client-ID).
    - DEMO: no user/org, but a client_id is provided (header or query param).
    - LOCAL: no headers; fall back to a single 'local' user/org.
    """

    roles = x_roles.split(",") if x_roles else []

    # Allow demo frontend to keep sending ?client_id=... for now
    query_client_id = request.query_params.get("client_id")
    client_id = x_client_id or query_client_id

    # --- Cloud mode: real auth in front of us ---
    if x_user_id or x_org_id:
        return RequestIdentity(
            user_id=x_user_id,
            org_id=x_org_id,
            roles=roles,
            client_id=client_id,  # optional; may be unused in cloud
            mode="cloud",
        )

    # --- Demo mode: no auth, but we have a client_id ---
    if client_id:
        return RequestIdentity(
            user_id="demo",
            org_id="demo",
            roles=["demo"],
            client_id=client_id,
            mode="demo",
        )

    # --- Local mode: dev / sidecar ---
    return RequestIdentity(
        user_id="local",
        org_id="local",
        roles=["dev"],
        client_id=None,
        mode="local",
    )
