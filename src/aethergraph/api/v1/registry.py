from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.api.v1.schemas.registry import (
    RegistryRegisterRequest,
    RegistryRegisterResponse,
)
from aethergraph.services.registry.registration_service import RegistrationService

router = APIRouter(tags=["registry"])


@router.post("/registry/register", response_model=RegistryRegisterResponse)
async def register_source(
    body: RegistryRegisterRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> RegistryRegisterResponse:
    reg = scoped_registry(identity)
    if reg.registration_service is None:
        raise HTTPException(status_code=503, detail="Registration service not configured")

    try:
        if body.source == "file":
            if not body.path:
                raise HTTPException(status_code=400, detail="path is required for source=file")
            result = await reg.register_by_file(
                body.path,
                app_config=body.app_config,
                agent_config=body.agent_config,
                persist=body.persist,
                strict=body.strict,
            )
        else:
            if not body.artifact_id and not body.uri:
                raise HTTPException(
                    status_code=400,
                    detail="artifact_id or uri is required for source=artifact",
                )
            result = await reg.register_by_artifact(
                artifact_id=body.artifact_id,
                uri=body.uri,
                app_config=body.app_config,
                agent_config=body.agent_config,
                persist=body.persist,
                strict=body.strict,
            )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    payload = RegistrationService.to_dict(result)
    return RegistryRegisterResponse(**payload)
