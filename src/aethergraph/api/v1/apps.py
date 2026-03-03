# aethergraph/api/v1/apps.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas import AppDescriptor
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.scope.scope import Scope

router = APIRouter(tags=["apps"])


def _scoped_registry(identity: RequestIdentity) -> RegistryFacade:
    reg = current_registry()
    return RegistryFacade(
        registry=reg,
        scope=Scope(
            org_id=identity.org_id,
            user_id=identity.user_id,
            client_id=identity.client_id,
            mode=identity.mode,
        ),
    )


def _ensure_delete_identity(identity: RequestIdentity) -> None:
    if identity.mode == "local":
        raise HTTPException(
            status_code=403,
            detail="Deleting apps requires authenticated tenant identity.",
        )
    if not (identity.org_id or identity.user_id or identity.client_id):
        raise HTTPException(
            status_code=403,
            detail="Missing tenant identity. Cannot delete app.",
        )


@router.get("/apps", response_model=list[AppDescriptor])
async def list_apps(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> list[AppDescriptor]:
    """
    List all registered apps.

    Each app is a graph (or graphfn) that has been decorated with `as_app={...}`.
    """
    reg = _scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    # {'app:metalens': '0.1.0', ...}
    entries = reg.list_apps(include_global=True)
    out: list[AppDescriptor] = []

    for ref, _version in entries.items():
        # ref is "app:<name>"
        try:
            _, name = ref.split(":", 1)
        except ValueError:
            # Defensive: ignore malformed keys
            continue

        meta = reg.get_meta(nspace="app", name=name, include_global=True) or {}
        scoped_meta = reg.get_meta(nspace="app", name=name, include_global=False)
        app_id = meta.get("id", name)
        graph_id = meta.get("graph_id", name)

        out.append(
            AppDescriptor(
                id=app_id,
                graph_id=graph_id,
                deletable=bool(scoped_meta),
                slash_commands=meta.get("slash_commands") or [],
                meta=meta,
            )
        )

    return out


@router.get("/apps/{app_id}", response_model=AppDescriptor)
async def get_app(
    app_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> AppDescriptor:
    reg = _scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    # Resolve by app id (we store app_id as the registry `name`)
    meta = reg.get_meta(nspace="app", name=app_id, include_global=True)
    if not meta:
        raise HTTPException(status_code=404, detail=f"App not found: {app_id}")

    graph_id = meta.get("graph_id", meta.get("backing", {}).get("name", app_id))
    scoped_meta = reg.get_meta(nspace="app", name=app_id, include_global=False)

    return AppDescriptor(
        id=meta.get("id", app_id),
        graph_id=graph_id,
        deletable=bool(scoped_meta),
        slash_commands=meta.get("slash_commands") or [],
        meta=meta,
    )


@router.delete("/apps/{app_id}")
async def delete_app(
    app_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> dict[str, str | bool]:
    _ensure_delete_identity(identity)
    reg = _scoped_registry(identity)

    scoped_meta = reg.get_meta(
        nspace="app",
        name=app_id,
        include_global=False,
    )
    if not scoped_meta:
        # Exists globally (or under another tenant), but not owned by this identity.
        any_meta = reg.get_meta(nspace="app", name=app_id, include_global=True)
        if any_meta:
            raise HTTPException(
                status_code=403,
                detail="App exists but cannot be deleted by this identity.",
            )
        raise HTTPException(status_code=404, detail=f"App not found: {app_id}")

    reg.unregister(nspace="app", name=app_id)
    return {"ok": True, "id": app_id}
