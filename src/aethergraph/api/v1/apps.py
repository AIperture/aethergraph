# aethergraph/api/v1/apps.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.input_schema import merge_input_schema_overrides, resolve_graph_input_schema
from aethergraph.api.v1.registry_helpers import ensure_delete_identity, scoped_registry
from aethergraph.api.v1.schemas.registry import AppDescriptor

router = APIRouter(tags=["apps"])


@router.get("/apps", response_model=list[AppDescriptor])
async def list_apps(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> list[AppDescriptor]:
    """
    List all registered apps.

    Each app is a graph (or graphfn) that has been decorated with `as_app={...}`.
    """
    reg = scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    # {'app:metalens': '0.1.0', ...}
    entries = reg.list_apps(include_global=True)
    out: list[AppDescriptor] = []

    print(f"🍎 Found {len(entries)} registered apps for tenant={identity} (including global)")
    for ref, _version in entries.items():
        # ref is "app:<name>"
        try:
            _, name = ref.split(":", 1)
        except ValueError:
            # Defensive: ignore malformed keys
            continue

        # `include_global=True` returns metadata from either:
        # - caller scope, or
        # - global scope (including built-ins)
        # Use this for display data.
        meta = reg.get_meta(nspace="app", name=name, include_global=True) or {}

        # `include_global=False` returns metadata only from caller scope.
        # If present, this app is caller-owned and can be deleted.
        scoped_meta = reg.get_meta(nspace="app", name=name, include_global=False)
        app_id = meta.get("id", name)
        graph_id = meta.get("graph_id", name)
        input_schema = merge_input_schema_overrides(
            resolve_graph_input_schema(reg, graph_id=graph_id),
            app_meta=meta,
        )

        out.append(
            AppDescriptor(
                id=app_id,
                graph_id=graph_id,
                deletable=bool(scoped_meta),
                slash_commands=meta.get("slash_commands") or [],
                input_schema=input_schema,
                meta=meta,
            )
        )

    return out


@router.get("/apps/{app_id}", response_model=AppDescriptor)
async def get_app(
    app_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> AppDescriptor:
    reg = scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    # Resolve by app id (we store app_id as the registry `name`)
    meta = reg.get_meta(nspace="app", name=app_id, include_global=True)
    if not meta:
        raise HTTPException(status_code=404, detail=f"App not found: {app_id}")

    graph_id = meta.get("graph_id", meta.get("backing", {}).get("name", app_id))
    input_schema = merge_input_schema_overrides(
        resolve_graph_input_schema(reg, graph_id=graph_id),
        app_meta=meta,
    )
    # If metadata exists in caller scope (not just global), allow delete UI.
    scoped_meta = reg.get_meta(nspace="app", name=app_id, include_global=False)

    return AppDescriptor(
        id=meta.get("id", app_id),
        graph_id=graph_id,
        deletable=bool(scoped_meta),
        slash_commands=meta.get("slash_commands") or [],
        input_schema=input_schema,
        meta=meta,
    )


@router.delete("/apps/{app_id}")
async def delete_app(
    app_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> dict[str, str | bool | int]:
    ensure_delete_identity(identity, "apps")
    reg = scoped_registry(identity)

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

    result = await reg.delete_registered_app(app_id=app_id)
    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to delete app '{app_id}': "
                + ("; ".join(result.errors) if result.errors else "unknown error")
            ),
        )
    return {"ok": True, "id": app_id, "removed_entries": result.removed_entries}
