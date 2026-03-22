from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response  # type: ignore
from pydantic import BaseModel, Field

from aethergraph.api.v1.deps import RequestIdentity, get_authn, get_identity, require_admin_key
from aethergraph.services.auth.authn import AuthnService, DemoGrant

router = APIRouter(prefix="/auth", tags=["auth"])


class InviteRedeemRequest(BaseModel):
    code: str = Field(..., min_length=1)


class InviteCreateRequest(BaseModel):
    grant_id: str = Field(..., min_length=1)
    org_id: str = Field(..., min_length=1)
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_agents: list[str] = Field(default_factory=list)
    client_label: str | None = None
    read_only: bool = False
    max_uses: int | None = None
    expires_in_hours: int = 24 * 7
    code: str | None = None  # custom invite code; if omitted, a random one is generated


class InviteCreateResponse(BaseModel):
    code: str
    grant_id: str
    max_uses: int | None
    expires_at: str | None


class AuthCapabilities(BaseModel):
    can_sign_in: bool = False
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
            demo_read_only=bool(grant.read_only) if grant else False,
        ),
    )


@router.post(
    "/invite/create",
    response_model=InviteCreateResponse,
    dependencies=[Depends(require_admin_key)],
)
async def invite_create(
    body: InviteCreateRequest,
    authn: AuthnService = Depends(get_authn),  # noqa: B008
) -> InviteCreateResponse:
    grant = DemoGrant(
        grant_id=body.grant_id,
        org_id=body.org_id,
        allowed_apps=body.allowed_apps,
        allowed_agents=body.allowed_agents,
        client_label=body.client_label,
        read_only=body.read_only,
    )
    invite = authn.create_invite_code(
        grant,
        max_uses=body.max_uses,
        expires_in_seconds=body.expires_in_hours * 3600,
        code=body.code,
    )
    return InviteCreateResponse(
        code=invite.code,
        grant_id=grant.grant_id,
        max_uses=invite.max_uses,
        expires_at=invite.expires_at.isoformat() if invite.expires_at else None,
    )


@router.post("/invite/redeem", response_model=AuthMeResponse)
async def invite_redeem(
    body: InviteRedeemRequest,
    request: Request,
    response: Response,
    authn: AuthnService = Depends(get_authn),  # noqa: B008
) -> AuthMeResponse:
    client_id = request.headers.get("X-Client-ID") or request.query_params.get("client_id")
    try:
        sess = authn.redeem_invite_code(body.code, client_id=client_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    grant = authn.get_grant(sess.grant_id)
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
                "apps": grant.allowed_apps if grant else [],
                "agents": grant.allowed_agents if grant else [],
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
