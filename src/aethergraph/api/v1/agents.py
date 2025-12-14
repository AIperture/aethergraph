# aethergraph/api/v1/agents.py

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas import AgentDescriptor
from aethergraph.core.runtime.runtime_registry import current_registry

router = APIRouter(tags=["agents"])


@router.get("/agents", response_model=list[AgentDescriptor])
async def list_agents(
    identity: Annotated[RequestIdentity, Depends(get_identity)] = None,
) -> list[AgentDescriptor]:
    """
    List all registered agents.

    These come from `as_agent={...}` (or legacy `agent="..."`) in your decorators.
    """
    reg = current_registry()
    if reg is None:
        raise HTTPException(status_code=500, detail="Registry not available")

    entries = reg.list_agents()  # {'agent:designer': '0.1.0', ...}
    out: list[AgentDescriptor] = []

    for ref, _version in entries.items():
        try:
            _, name = ref.split(":", 1)
        except ValueError:
            continue

        meta = reg.get_meta(nspace="agent", name=name) or {}
        agent_id = meta.get("id", name)

        out.append(
            AgentDescriptor(
                id=agent_id,
                meta=meta,
            )
        )

    return out
