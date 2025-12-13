from typing import Literal

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.auth.authz import AuthZService


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
        return (self.org_id, self.user_id)


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
        # Treat client_id as the actual user_id for demo
        demo_user_id = f"demo:{client_id}"
        return RequestIdentity(
            user_id=demo_user_id,
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


def _rate_key(identity: RequestIdentity) -> str:
    """
    Compute a stable key for rate limiting.

    - CLOUD: prefer org_id, then user_id
    - DEMO: use client_id if present, else "demo"
    - LOCAL: just "local"
    """
    if identity.mode == "cloud":
        return identity.org_id or identity.user_id or "anonymous"

    if identity.mode == "demo":
        # Each browser/client gets its own key if possible
        return identity.client_id or "demo"

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

    # In local/dev mode, don't annoy you with limits
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
