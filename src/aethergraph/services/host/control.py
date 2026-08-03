"""Authenticated local supervisor control surface for AG Host."""

from __future__ import annotations

from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict


class _ControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HostHealth(_ControlModel):
    status: str
    deployment_id: str
    build_id: str
    manifest_digest: str


class HostReadiness(HostHealth):
    ready: bool


class HostDiagnostics(HostReadiness):
    workspace_identity: str
    providers: tuple[dict[str, str | None], ...]


def install_host_control_routes(*, app, host, control_token: str) -> None:
    """Install authenticated health, readiness, and diagnostic endpoints.

    The per-launch token remains in a route closure and every response is
    bounded to immutable identities and redacted provider state.

    Examples:
        Install routes on a composed Host application:
            ```python
            install_host_control_routes(
                app=app,
                host=host,
                control_token=launch_token,
            )
            ```

        Probe readiness with the launch token:
            ```python
            response = client.get(
                "/_host/ready",
                headers={"X-AG-Host-Control": launch_token},
            )
            ```

    Args:
        app: FastAPI application owned by the composed Host.
        host: Fully composed immutable `AGHost` instance.
        control_token: High-entropy per-launch supervisor token.

    Returns:
        None: Routes are added directly to `app`.

    Notes:
        Provider exceptions, credentials, settings, and filesystem paths are
        never returned by this control surface.
    """

    if len(control_token) < 32:
        raise ValueError("AG Host control token must contain at least 32 characters.")
    router = APIRouter(prefix="/_host", include_in_schema=False)

    def authorize(
        supplied: Annotated[str | None, Header(alias="X-AG-Host-Control")] = None,
    ) -> None:
        if supplied is None or not compare_digest(supplied, control_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    def health_payload() -> HostHealth:
        return HostHealth(
            status="alive",
            deployment_id=host.manifest.deployment_id,
            build_id=host.manifest.build_id,
            manifest_digest=host.manifest.manifest_digest,
        )

    @router.get("/health", response_model=HostHealth)
    async def health(
        supplied: Annotated[str | None, Header(alias="X-AG-Host-Control")] = None,
    ) -> HostHealth:
        authorize(supplied)
        return health_payload()

    @router.get("/ready", response_model=HostReadiness)
    async def ready(
        response: Response,
        supplied: Annotated[str | None, Header(alias="X-AG-Host-Control")] = None,
    ) -> HostReadiness:
        authorize(supplied)
        is_ready = host.integration_manager.ready
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HostReadiness(
            **health_payload().model_dump(),
            ready=is_ready,
        )

    @router.get("/diagnostics", response_model=HostDiagnostics)
    async def diagnostics(
        response: Response,
        supplied: Annotated[str | None, Header(alias="X-AG-Host-Control")] = None,
    ) -> HostDiagnostics:
        authorize(supplied)
        is_ready = host.integration_manager.ready
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        providers = tuple(
            {
                "integration_id": item.integration_id,
                "integration_kind": item.integration_kind.value,
                "state": item.state.value,
                "error_code": item.error_code,
            }
            for item in host.integration_manager.statuses()
        )
        return HostDiagnostics(
            **health_payload().model_dump(),
            ready=is_ready,
            workspace_identity=host.manifest.workspace_identity,
            providers=providers,
        )

    app.include_router(router)


__all__ = [
    "HostDiagnostics",
    "HostHealth",
    "HostReadiness",
    "install_host_control_routes",
]
