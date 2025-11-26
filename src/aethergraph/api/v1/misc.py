# /health /config


from fastapi import APIRouter, Depends

from .deps import RequestIdentity, get_identity
from .schemas import ConfigLLMProvider, ConfigResponse, HealthResponse

router = APIRouter(tags=["misc"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Simple health check endpoint.
    """
    # TODO: optionally include deeper checks (DB, Redis, etc.)
    return HealthResponse(status="ok", version="0.1.0a1")


@router.get("/config", response_model=ConfigResponse)
async def config_info(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ConfigResponse:
    """
    Return sanitized config info that's safe for UI.

    TODO:
      - Read from AppSettings.
      - Mask secrets; only expose high-level info.
    """
    # Stub example
    return ConfigResponse(
        version="0.1.0a1",
        storage_backends={
            "memory": "fs_jsonl",
            "artifacts": "fs",
        },
        llm_providers=[
            ConfigLLMProvider(name="openai", model="gpt-4o-mini", enabled=True),
        ],
        features={
            "ws_channels": True,
            "artifact_search": True,
            "memory_search": True,
        },
    )
