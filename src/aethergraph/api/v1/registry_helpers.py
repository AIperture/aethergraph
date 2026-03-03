from fastapi import HTTPException  # type: ignore

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.registry.facade import RegistryFacade
from aethergraph.services.scope.scope import Scope


def scoped_registry(identity: RequestIdentity) -> RegistryFacade:
    registration_service = None
    try:
        registration_service = getattr(current_services(), "registration_service", None)
    except Exception:
        registration_service = None

    return RegistryFacade(
        registry=current_registry(),
        scope=Scope(
            org_id=identity.org_id,
            user_id=identity.user_id,
            client_id=identity.client_id,
            mode=identity.mode,
        ),
        registration_service=registration_service,
    )


def ensure_delete_identity(identity: RequestIdentity, resource_name: str) -> None:
    if identity.mode == "local":
        raise HTTPException(
            status_code=403,
            detail=f"Deleting {resource_name} requires authenticated tenant identity.",
        )
    if not (identity.org_id or identity.user_id or identity.client_id):
        raise HTTPException(
            status_code=403,
            detail=f"Missing tenant identity. Cannot delete {resource_name}.",
        )
