# aethergraph/api/v1/apps.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.api.v1.deps import (
    RequestIdentity,
    enforce_run_rate_limits,
    get_identity,
    require_runs_execute,
)
from aethergraph.api.v1.input_schema import merge_input_schema_overrides, resolve_graph_input_schema
from aethergraph.api.v1.registry_helpers import ensure_delete_identity, scoped_registry
from aethergraph.api.v1.schemas.registry import AppDescriptor
from aethergraph.api.v1.schemas.runs import RunCreateRequest, RunCreateResponse
from aethergraph.core.runtime.run_manager import DuplicateRunIdError, RunManager
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.core.runtime.runtime_services import current_services

router = APIRouter(tags=["apps"])


def _resolve_app_graph_id(app_meta: dict, *, default: str | None = None) -> str | None:
    graph_id = app_meta.get("graph_id")
    if graph_id:
        return str(graph_id)

    backing = app_meta.get("backing")
    if isinstance(backing, dict):
        backing_name = backing.get("name")
        if backing_name:
            return str(backing_name)

    graph_name = app_meta.get("graph_name")
    if graph_name:
        return str(graph_name)

    flow_id = app_meta.get("flow_id")
    if flow_id:
        return str(flow_id)

    if default:
        return str(default)
    return None


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
        graph_id = _resolve_app_graph_id(meta, default=name) or name
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

    graph_id = _resolve_app_graph_id(meta, default=app_id)
    if not graph_id:
        raise HTTPException(status_code=400, detail=f"App '{app_id}' has no backing graph_id")
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


@router.post(
    "/apps/{app_id}/runs",
    response_model=RunCreateResponse,
    dependencies=[Depends(enforce_run_rate_limits)],  # noqa: B008
)
async def create_app_run(
    app_id: str,
    body: RunCreateRequest,
    identity: RequestIdentity = Depends(require_runs_execute),  # noqa: B008
) -> RunCreateResponse:
    container = current_services()
    rm: RunManager = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    reg = scoped_registry(identity)
    app_meta = reg.get_meta(nspace="app", name=app_id, include_global=True)
    if not app_meta:
        raise HTTPException(status_code=404, detail=f"App not found: {app_id}")

    graph_id = _resolve_app_graph_id(app_meta)
    if not graph_id:
        raise HTTPException(status_code=400, detail=f"App '{app_id}' has no backing graph_id")

    app_vis = app_meta.get("run_visibility")
    app_imp = app_meta.get("run_importance")
    app_vis = RunVisibility(app_vis) if app_vis else None
    app_imp = RunImportance(app_imp) if app_imp else None
    try:
        record = await rm.submit_run(
            graph_id=graph_id,
            inputs=body.inputs or {},
            run_id=body.run_id,
            tags=body.tags,
            session_id=body.session_id,
            identity=identity,
            origin=body.origin or RunOrigin.app,
            visibility=body.visibility or app_vis or RunVisibility.normal,
            importance=body.importance or app_imp or RunImportance.normal,
            agent_id=body.agent_id or None,
            app_id=app_id,
            app_name=body.app_name or app_meta.get("name") or app_id,
            run_config=body.run_config or {},
        )
    except DuplicateRunIdError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    return RunCreateResponse(
        run_id=record.run_id,
        graph_id=record.graph_id,
        status=record.status,
        outputs=None,
        has_waits=False,
        continuations=[],
        started_at=record.started_at,
        finished_at=record.finished_at,
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
