from __future__ import annotations

from typing import Any, Literal

from fastapi import Depends, Header, HTTPException, Request, status  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.auth.authn import AuthnService, DemoGrant
from aethergraph.services.auth.authz import AuthZService
from aethergraph.services.scope.tenant import registry_tenant_from_identity


class RequestIdentity(BaseModel):
    user_id: str | None = None
    org_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    client_id: str | None = None
    grant_id: str | None = None
    auth_source: str | None = None
    catalog_scope: dict[str, list[str]] | None = None
    mode: Literal["cloud", "demo", "local"] = "local"

    @property
    def is_cloud(self) -> bool:
        return self.mode == "cloud"

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    @property
    def tenant_key(self) -> tuple[str | None, str | None]:
        tenant = registry_tenant_from_identity(self)
        if tenant is None:
            return (None, None)
        return (tenant.get("org_id"), tenant.get("user_id"))


def _get_deploy_mode() -> str:
    try:
        container = current_services()
        settings = getattr(container, "settings", None)
        if settings is not None:
            return getattr(settings, "deploy_mode", "local") or "local"
    except Exception:
        pass
    return "local"


def get_authn() -> AuthnService:
    container = current_services()
    return container.authn  # type: ignore[return-value]


async def require_admin_key(request: Request) -> None:
    """Gate admin endpoints behind an API key when configured."""
    container = current_services()
    settings = getattr(container, "settings", None)
    key = getattr(settings, "auth", None) and settings.auth.admin_api_key
    if key is None:
        return  # No key configured = open access (local dev)
    provided = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not provided or provided != key.get_secret_value():
        raise HTTPException(status_code=403, detail="Admin API key required")


def _catalog_scope_for_grant(grant: DemoGrant | None) -> dict[str, list[str]] | None:
    if grant is None:
        return None
    scope: dict[str, list[str]] = {}
    if grant.allowed_apps:
        scope["apps"] = list(grant.allowed_apps)
    if grant.allowed_agents:
        scope["agents"] = list(grant.allowed_agents)
    return scope or None


async def get_identity(
    request: Request,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    x_org_id: str | None = Header(None, alias="X-Org-ID"),
    x_roles: str | None = Header(None, alias="X-Roles"),
    x_client_id: str | None = Header(None, alias="X-Client-ID"),
    x_mode: str | None = Header(None, alias="X-Mode"),
) -> RequestIdentity:
    deploy_mode = _get_deploy_mode()
    roles = x_roles.split(",") if x_roles else []
    client_id = x_client_id or request.query_params.get("client_id")
    authn = get_authn()
    resolved = authn.resolve(
        deploy_mode=deploy_mode,
        session_id=request.cookies.get(authn.cookie_name),
        client_id=client_id,
        x_user_id=x_user_id,
        x_org_id=x_org_id,
        roles=roles,
        x_mode=x_mode,
    )
    if resolved.mode == "cloud_proxy":
        return RequestIdentity(
            user_id=resolved.user_id,
            org_id=resolved.org_id,
            roles=resolved.roles,
            client_id=resolved.client_id,
            mode="cloud",
            auth_source=resolved.auth_source,
        )
    if resolved.mode == "demo_guest":
        return RequestIdentity(
            user_id=resolved.user_id,
            org_id=resolved.org_id,
            roles=resolved.roles,
            client_id=resolved.client_id,
            mode="demo",
            grant_id=resolved.session.grant_id if resolved.session else None,
            auth_source=resolved.auth_source,
            catalog_scope=_catalog_scope_for_grant(resolved.grant),
        )
    return RequestIdentity(
        user_id=resolved.user_id,
        org_id=resolved.org_id,
        roles=resolved.roles,
        client_id=resolved.client_id,
        mode="local",
        auth_source=resolved.auth_source,
    )


def _rate_key(identity: RequestIdentity) -> str:
    if identity.mode == "cloud":
        return identity.org_id or identity.user_id or "anonymous"
    if identity.mode == "demo":
        return identity.user_id or "demo"
    return "local"


def get_authz() -> AuthZService:
    container = current_services()
    return container.authz  # type: ignore[return-value]


def catalog_allows(
    identity: RequestIdentity, kind: Literal["apps", "agents"], item_id: str
) -> bool:
    scope = getattr(identity, "catalog_scope", None) or {}
    allowed = scope.get(kind)
    if not allowed:
        return True
    return item_id in allowed


def ensure_identity_matches_owner(
    identity: RequestIdentity,
    *,
    user_id: str | None,
    org_id: str | None,
    missing_status: int = 404,
    missing_detail: str = "Resource not found",
) -> None:
    if identity.mode == "local":
        return
    if identity.user_id is None:
        raise HTTPException(status_code=403, detail="User identity required")
    if user_id is not None and user_id != identity.user_id:
        raise HTTPException(status_code=missing_status, detail=missing_detail)
    if identity.org_id and org_id is not None and org_id != identity.org_id:
        raise HTTPException(status_code=missing_status, detail=missing_detail)


def artifact_belongs_to_identity(identity: RequestIdentity, artifact: Any) -> bool:
    if identity.mode == "local":
        return True
    labels = getattr(artifact, "labels", None) or {}
    art_user = labels.get("user_id")
    art_org = labels.get("org_id")
    if identity.user_id and art_user and art_user != identity.user_id:
        return False
    if identity.org_id and art_org and art_org != identity.org_id:
        return False
    return bool(art_user or art_org)


async def require_runs_execute(
    identity: RequestIdentity = Depends(get_identity),  # noqa B008
) -> RequestIdentity:
    container = current_services()
    if container.authz:
        await container.authz.authorize(identity=identity, scope="runs", action="execute")
    return identity


async def enforce_run_rate_limits(
    request: Request,
    identity: RequestIdentity = Depends(get_identity),  # noqa B008
) -> None:
    container = current_services()
    settings = getattr(container, "settings", None)
    if not settings or not settings.rate_limit.enabled:
        return
    if identity.mode == "local":
        return
    rl_cfg = settings.rate_limit
    meter = getattr(container, "metering", None)
    if meter is not None:
        overview = await meter.get_overview(
            user_id=identity.user_id,
            org_id=identity.org_id,
            window=rl_cfg.runs_window,
        )
        if overview.get("runs", 0) >= rl_cfg.max_runs_per_window:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Run limit exceeded: at most "
                    f"{rl_cfg.max_runs_per_window} runs per {rl_cfg.runs_window}."
                ),
            )
    limiter = getattr(container, "run_burst_limiter", None)
    if limiter is not None:
        key = _rate_key(identity)
        if not limiter.allow(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many runs started in a short period. Please wait a moment.",
            )
