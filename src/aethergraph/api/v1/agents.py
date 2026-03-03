# aethergraph/api/v1/agents.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas import AgentDescriptor
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.scope.scope import Scope

router = APIRouter(tags=["agents"])


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
            detail="Deleting agents requires authenticated tenant identity.",
        )
    if not (identity.org_id or identity.user_id or identity.client_id):
        raise HTTPException(
            status_code=403,
            detail="Missing tenant identity. Cannot delete agent.",
        )


@router.get("/agents", response_model=list[AgentDescriptor])
async def list_agents(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> list[AgentDescriptor]:
    """
    List all registered agents.

    These come from `as_agent={...}` (or legacy `agent="..."`) in your decorators.
    """
    reg = _scoped_registry(identity)
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
        agent_id = meta.get("id", name)

        out.append(
            AgentDescriptor(
                id=agent_id,
                graph_id=meta.get("graph_id", name),
                meta=meta,
            )
        )

    return out


@router.get("/agents/{agent_id}", response_model=AgentDescriptor)
async def get_agent(
    agent_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> AgentDescriptor:
    reg = _scoped_registry(identity)
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    meta = reg.get_meta(nspace="agent", name=agent_id, include_global=True)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    graph_id = meta.get("graph_id", meta.get("backing", {}).get("name", agent_id))
    return AgentDescriptor(id=meta.get("id", agent_id), graph_id=graph_id, meta=meta)


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> dict[str, str | bool]:
    _ensure_delete_identity(identity)
    reg = _scoped_registry(identity)

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

    reg.unregister(nspace="agent", name=agent_id)
    return {"ok": True, "id": agent_id}
