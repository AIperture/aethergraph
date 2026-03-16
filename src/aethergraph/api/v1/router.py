from fastapi import APIRouter  # type: ignore

from .agents import router as agents_router
from .apps import router as apps_router
from .artifacts import router as artifacts_router
from .channels import router as channels_router
from .graphs import router as graphs_router
from .identity import router as identity_router
from .inspect import router as inspect_router
from .memory import router as memory_router
from .misc import router as misc_router
from .registry import router as registry_router
from .runs import router as runs_router
from .session import router as session_router
from .settings import router as settings_router
from .stats import router as stats_router
from .triggers import router as triggers_router
from .viz import router as viz_router

router = APIRouter()
router.include_router(runs_router)
router.include_router(graphs_router)
router.include_router(artifacts_router)
router.include_router(channels_router)
router.include_router(memory_router)
router.include_router(inspect_router)
router.include_router(stats_router)
router.include_router(identity_router)
router.include_router(misc_router)
router.include_router(viz_router)
router.include_router(session_router)
router.include_router(apps_router)
router.include_router(agents_router)
router.include_router(triggers_router)
router.include_router(settings_router)
router.include_router(registry_router)
