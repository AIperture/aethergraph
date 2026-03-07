from typing import Literal

from fastapi import Depends, Header, HTTPException, Request, status  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.auth.authz import AuthZService
from aethergraph.services.scope.tenant import registry_tenant_from_identity


class RequestIdentity(BaseModel):
    user_id: str | None = None
    org_id: str | None = None
    roles: list[str] = Field(default_factory=list)

    # Demo-only/browser identity
    client_id: str | None = None

    # How this request is “authenticated”
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
        """Convenience key for tenant scoping."""
        tenant = registry_tenant_from_identity(self)
        if tenant is None:
            return (None, None)
        return (tenant.get("org_id"), tenant.get("user_id"))


def _get_deploy_mode() -> str:
    """Read server-level deploy_mode from settings. Falls back to 'local'."""
    try:
        container = current_services()
        settings = getattr(container, "settings", None)
        if settings is not None:
            return getattr(settings, "deploy_mode", "local") or "local"
    except Exception:
        pass
    return "local"


async def get_identity(
    request: Request,
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    x_org_id: str | None = Header(None, alias="X-Org-ID"),
    x_roles: str | None = Header(None, alias="X-Roles"),
    x_client_id: str | None = Header(None, alias="X-Client-ID"),
    x_mode: str | None = Header(None, alias="X-Mode"),
) -> RequestIdentity:
    """
    Identity extraction hook.

    Behaviour is driven by the server-level ``deploy_mode`` setting
    (env: AETHERGRAPH_DEPLOY_MODE) **and** per-request headers.

    deploy_mode="local" (default / OSS):
        X-Client-ID is recorded for tracking but does NOT create a
        separate tenant scope. CLI, script, and UI all share one
        identity (user_id="local").

    deploy_mode="demo":
        X-Client-ID triggers demo-mode tenant isolation. Each browser
        gets user_id="demo:<client_id>", so different demo visitors
        only see their own runs.

    deploy_mode="cloud":
        Expects an auth gateway to inject X-User-ID / X-Org-ID.
        Falls back to demo if only X-Client-ID is present.

    A per-request X-Mode header can also force demo mode regardless
    of deploy_mode (useful for testing).
    """
    deploy_mode = _get_deploy_mode()

    roles = x_roles.split(",") if x_roles else []

    query_client_id = request.query_params.get("client_id")
    client_id = x_client_id or query_client_id

    # --- Cloud mode: real auth in front of us ---
    if x_user_id or x_org_id:
        return RequestIdentity(
            user_id=x_user_id,
            org_id=x_org_id,
            roles=roles,
            client_id=client_id,
            mode="cloud",
        )

    # --- Demo mode ---
    # Triggered when:
    #   1) deploy_mode="demo" and a client_id is present, OR
    #   2) per-request X-Mode: demo header is sent (any deploy_mode)
    if client_id and (deploy_mode == "demo" or x_mode == "demo"):
        demo_user_id = f"demo:{client_id}"
        return RequestIdentity(
            user_id=demo_user_id,
            org_id="demo",
            roles=["demo"],
            client_id=client_id,
            mode="demo",
        )

    # --- Local mode: dev / sidecar ---
    # All requests share a single identity. X-Client-ID is kept for
    # tracking but does not gate visibility of runs/artifacts.
    return RequestIdentity(
        user_id="local",
        org_id="local",
        roles=["dev"],
        client_id=client_id,
        mode="local",
    )


def _rate_key(identity: RequestIdentity) -> str:
    """
    Compute a stable key for rate limiting.

    - CLOUD: prefer org_id, then user_id
    - DEMO: use resolved user_id (demo:<client_id>) for consistency
    - LOCAL: just "local"
    """
    if identity.mode == "cloud":
        return identity.org_id or identity.user_id or "anonymous"

    if identity.mode == "demo":
        return identity.user_id or "demo"

    # local / dev
    return "local"


def get_authz() -> AuthZService:
    container = current_services()
    return container.authz  # type: ignore[return-value]


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

    # In local/dev mode, don't annoy with limits
    if identity.mode == "local":
        return

    rl_cfg = settings.rate_limit

    # ---------- 1) Long-window per-identity cap via metering ----------
    meter = getattr(container, "metering", None)
    if meter is not None:
        # For demo mode this will be user_id="demo", org_id="demo",
        # so all demo clients share the hourly cap. That's fine for now.
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

    # ---------- 2) Short-burst limiter (in-memory) ----------
    limiter = getattr(container, "run_burst_limiter", None)
    if limiter is not None:
        key = _rate_key(identity)
        if not limiter.allow(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many runs started in a short period. Please wait a moment.",
            )
