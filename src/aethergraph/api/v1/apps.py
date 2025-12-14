# aethergraph/api/v1/apps.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas import AppDescriptor
from aethergraph.core.runtime.runtime_registry import current_registry

router = APIRouter(tags=["apps"])


@router.get("/apps", response_model=list[AppDescriptor])
async def list_apps(
    identity: Annotated[RequestIdentity, Depends(get_identity)] = None,
) -> list[AppDescriptor]:
    """
    List all registered apps.

    Each app is a graph (or graphfn) that has been decorated with `as_app={...}`.
    """
    reg = current_registry()
    if reg is None:
        # You can also return [] silently; up to you
        raise HTTPException(status_code=500, detail="Registry not available")

    # {'app:metalens': '0.1.0', ...}
    entries = reg.list_apps()
    out: list[AppDescriptor] = []

    for ref, _version in entries.items():
        # ref is "app:<name>"
        try:
            _, name = ref.split(":", 1)
        except ValueError:
            # Defensive: ignore malformed keys
            continue

        meta = reg.get_meta(nspace="app", name=name) or {}
        app_id = meta.get("id", name)
        graph_id = meta.get("graph_id", name)

        out.append(
            AppDescriptor(
                id=app_id,
                graph_id=graph_id,
                meta=meta,
            )
        )

    return out
