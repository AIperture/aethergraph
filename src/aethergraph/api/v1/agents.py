# aethergraph/api/v1/agents.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity, catalog_allows, get_identity
from aethergraph.api.v1.registry_helpers import ensure_delete_identity, scoped_registry
from aethergraph.api.v1.schemas.registry import AgentDescriptor

router = APIRouter(tags=["agents"])


@router.get("/agents", response_model=list[AgentDescriptor])
async def list_agents(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> list[AgentDescriptor]:
    """
    List all registered agents.

    These come from `as_agent={...}` (or legacy `agent="..."`) in your decorators.
    """
    reg = scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    entries = reg.list_agents(include_global=True)  # {'agent:designer': '0.1.0', ...}
    out: list[AgentDescriptor] = []

    for ref, _version in entries.items():
        try:
            _, name = ref.split(":", 1)
        except ValueError:
            continue

        meta = reg.get_meta(nspace="agent", name=name, include_global=True) or {}
        # Only scope-owned (non-global) entries are deletable for this caller.
        scoped_meta = reg.get_meta(nspace="agent", name=name, include_global=False)
        agent_id = meta.get("id", name)
        if not catalog_allows(identity, "agents", agent_id):
            continue

        out.append(
            AgentDescriptor(
                id=agent_id,
                graph_id=meta.get("graph_id", name),
                deletable=bool(scoped_meta),
                slash_commands=meta.get("slash_commands") or [],
                meta=meta,
            )
        )

    return out


@router.get("/agents/{agent_id}", response_model=AgentDescriptor)
async def get_agent(
    agent_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> AgentDescriptor:
    reg = scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    meta = reg.get_meta(nspace="agent", name=agent_id, include_global=True)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    if not catalog_allows(identity, "agents", meta.get("id", agent_id)):
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    graph_id = meta.get("graph_id", meta.get("backing", {}).get("name", agent_id))
    # If metadata exists in caller scope (not just global), allow delete UI.
    scoped_meta = reg.get_meta(nspace="agent", name=agent_id, include_global=False)
    return AgentDescriptor(
        id=meta.get("id", agent_id),
        graph_id=graph_id,
        deletable=bool(scoped_meta),
        slash_commands=meta.get("slash_commands") or [],
        meta=meta,
    )


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> dict[str, str | bool | int]:
    ensure_delete_identity(identity, "agents")
    reg = scoped_registry(identity)

    scoped_meta = reg.get_meta(
        nspace="agent",
        name=agent_id,
        include_global=False,
    )
    if not scoped_meta:
        # Exists globally (or under another tenant), but not owned by this identity.
        any_meta = reg.get_meta(nspace="agent", name=agent_id, include_global=True)
        if any_meta:
            raise HTTPException(
                status_code=403,
                detail="Agent exists but cannot be deleted by this identity.",
            )
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    result = await reg.delete_registered_agent(agent_id=agent_id)
    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to delete agent '{agent_id}': "
                + ("; ".join(result.errors) if result.errors else "unknown error")
            ),
        )
    return {"ok": True, "id": agent_id, "removed_entries": result.removed_entries}
