"""Closed API route surface for immutable AG Host deployments."""

from fastapi import APIRouter

from .agent_endpoints import router as agent_endpoints_router
from .agents import router as agents_router
from .apps import router as apps_router
from .artifacts import router as artifacts_router
from .auth import router as auth_router
from .graphs import router as graphs_router
from .identity import router as identity_router
from .memory import router as memory_router
from .misc import router as misc_router
from .observability import router as observability_router
from .runs import router as runs_router
from .session import router as session_router
from .settings import router as settings_router
from .stats import router as stats_router
from .viz import router as viz_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(agent_endpoints_router)


def _include_read_routes(source: APIRouter) -> None:
    for route in source.routes:
        methods = getattr(route, "methods", None)
        if methods == {"GET"}:
            router.routes.append(route)


for read_router in (
    agents_router,
    apps_router,
    artifacts_router,
    graphs_router,
    identity_router,
    memory_router,
    misc_router,
    observability_router,
    runs_router,
    session_router,
    settings_router,
    stats_router,
    viz_router,
):
    _include_read_routes(read_router)


_SAFE_MUTATIONS = {
    ("POST", "/artifacts/{artifact_id}/pin"),
    ("POST", "/artifacts/search"),
    ("POST", "/memory/search"),
    ("PATCH", "/sessions/{session_id}"),
    ("DELETE", "/sessions/{session_id}"),
}
for mutable_router in (artifacts_router, memory_router, session_router):
    for route in mutable_router.routes:
        methods = getattr(route, "methods", set())
        if any((method, route.path) in _SAFE_MUTATIONS for method in methods):
            router.routes.append(route)


__all__ = ["router"]
