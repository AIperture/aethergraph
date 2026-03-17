from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response  # type: ignore
from pydantic import BaseModel, Field

from aethergraph.api.v1.deps import RequestIdentity, get_authn, get_identity
from aethergraph.services.auth.authn import AuthnService, DemoGrant

router = APIRouter(prefix="/auth", tags=["auth"])


class DemoExchangeRequest(BaseModel):
    token: str = Field(..., min_length=1)


class AuthCapabilities(BaseModel):
    can_sign_in: bool = False
    can_use_demo_links: bool = False
    demo_read_only: bool = False


class AuthMeResponse(BaseModel):
    authenticated: bool
    mode: str
    user_id: str | None
    org_id: str | None
    roles: list[str]
    client_id: str | None
    grant_id: str | None
    auth_source: str | None
    catalog_scope: dict[str, list[str]] | None = None
    client_label: str | None = None
    capabilities: AuthCapabilities


def _grant_for_identity(authn: AuthnService, identity: RequestIdentity) -> DemoGrant | None:
    return authn.get_grant(identity.grant_id)


def _build_me_response(authn: AuthnService, identity: RequestIdentity) -> AuthMeResponse:
    grant = _grant_for_identity(authn, identity)
    return AuthMeResponse(
        authenticated=identity.mode != "local",
        mode=identity.mode,
        user_id=identity.user_id,
        org_id=identity.org_id,
        roles=identity.roles,
        client_id=identity.client_id,
        grant_id=identity.grant_id,
        auth_source=identity.auth_source,
        catalog_scope=identity.catalog_scope,
        client_label=grant.client_label if grant else None,
        capabilities=AuthCapabilities(
            can_sign_in=False,
            can_use_demo_links=True,
            demo_read_only=bool(grant.read_only) if grant else False,
        ),
    )


@router.post("/demo/exchange", response_model=AuthMeResponse)
async def demo_exchange(
    body: DemoExchangeRequest,
    request: Request,
    response: Response,
    authn: AuthnService = Depends(get_authn),  # noqa: B008
) -> AuthMeResponse:
    try:
        grant = authn.parse_demo_token(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    client_id = request.headers.get("X-Client-ID") or request.query_params.get("client_id")
    sess = authn.create_demo_session(grant=grant, client_id=client_id)
    response.set_cookie(
        key=authn.cookie_name,
        value=sess.session_id,
        httponly=True,
        secure=authn.cookie_secure,
        samesite=authn.cookie_samesite,
        max_age=authn.session_ttl_seconds,
        path="/",
    )
    identity = RequestIdentity(
        user_id=sess.user_id,
        org_id=sess.org_id,
        roles=sess.roles,
        client_id=client_id or sess.client_id,
        grant_id=sess.grant_id,
        auth_source="demo_guest_session",
        catalog_scope={
            k: v
            for k, v in {
                "apps": grant.allowed_apps,
                "agents": grant.allowed_agents,
            }.items()
            if v
        }
        or None,
        mode="demo",
    )
    return _build_me_response(authn, identity)


@router.get("/me", response_model=AuthMeResponse)
async def auth_me(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
    authn: AuthnService = Depends(get_authn),  # noqa: B008
) -> AuthMeResponse:
    return _build_me_response(authn, identity)


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    authn: AuthnService = Depends(get_authn),  # noqa: B008
) -> dict[str, bool]:
    authn.delete_session(request.cookies.get(authn.cookie_name))
    response.delete_cookie(key=authn.cookie_name, path="/")
    return {"ok": True}
