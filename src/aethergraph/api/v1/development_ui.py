"""Local development bootstrap for the canonical AG UI endpoint protocol."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["development-ui"])


@router.get("/ui/bootstrap")
async def development_ui_bootstrap(request: Request) -> dict[str, Any]:
    """Return the process-local AG UI endpoint catalog.

    The mutable development sidecar exposes this catalog so an independently
    served Vite UI can select canonical agent endpoints without a supervisor
    launch URL.

    Examples:
        Discover the default development endpoint:
            ```python
            GET /api/v1/ui/bootstrap
            ```

        Select the endpoint for an agent:
            ```python
            endpoint = response["endpoints"][0]["endpoint_id"]
            ```

    Args:
        request: FastAPI request carrying development bootstrap state.

    Returns:
        dict[str, Any]: Development mode, default agent, and endpoint catalog.

    Notes:
        Immutable deployment applications do not install this router.
    """

    bootstrap = getattr(request.app.state, "development_ui_bootstrap", None)
    if bootstrap is None:
        raise HTTPException(status_code=404, detail="Development AG UI is unavailable.")
    return bootstrap
