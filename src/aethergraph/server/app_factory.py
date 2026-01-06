import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from aethergraph.api.v1.agents import router as agents_router
from aethergraph.api.v1.apps import router as apps_router
from aethergraph.api.v1.artifacts import router as artifacts_router
from aethergraph.api.v1.graphs import router as graphs_router
from aethergraph.api.v1.identity import router as identity_router
from aethergraph.api.v1.memory import router as memory_router
from aethergraph.api.v1.misc import router as misc_router
from aethergraph.api.v1.runs import router as runs_router
from aethergraph.api.v1.session import router as session_router
from aethergraph.api.v1.stats import router as stats_router
from aethergraph.api.v1.viz import router as vis_router

# include apis
from aethergraph.config.config import AppSettings
from aethergraph.core.runtime.runtime_services import install_services

# import built-in agents and plugins to register them
from aethergraph.plugins.agents.default_chat_agent import *  # noqa: F403

# channel routes
from aethergraph.server.loading import GraphLoader, LoadSpec
from aethergraph.services.container.default_container import build_default_container
from aethergraph.utils.optdeps import require

logger = logging.getLogger(__name__)


def create_app(
    *,
    workspace: str = "./aethergraph_data",
    cfg: Optional["AppSettings"] = None,
    log_level: str = "info",
) -> FastAPI:
    """
    Builds the FastAPI app, registers routers, and installs all services
    into app.state.container (and globally via install_services()).
    """

    # Resolve settings and container up front so lifespan can capture them
    settings = cfg or AppSettings()
    settings.logging.level = log_level

    container = build_default_container(root=workspace, cfg=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- Startup: attach settings/container and start external transports ---
        app.state.settings = settings
        app.state.container = container

        slack_task = None
        tg_task = None

        # Slack Socket Mode
        slack_cfg = settings.slack
        if (
            slack_cfg
            and slack_cfg.enabled
            and slack_cfg.socket_mode_enabled
            and slack_cfg.bot_token
            and slack_cfg.app_token
        ):
            require("slack_sdk", "slack")
            from ..plugins.channel.websockets.slack_ws import SlackSocketModeRunner

            runner = SlackSocketModeRunner(container=container, settings=settings)
            app.state.slack_socket_runner = runner
            slack_task = asyncio.create_task(runner.start())

        # Telegram polling
        tg_cfg = settings.telegram
        if tg_cfg and tg_cfg.enabled and tg_cfg.polling_enabled and tg_cfg.bot_token:
            from ..plugins.channel.websockets.telegram_polling import TelegramPollingRunner

            tg_runner = TelegramPollingRunner(container=container, settings=settings)
            app.state.telegram_polling_runner = tg_runner
            tg_task = asyncio.create_task(tg_runner.start())

        try:
            # Hand control back to FastAPI / TestClient
            yield
        finally:
            # --- Shutdown: best-effort cleanup of background tasks ---
            for task in (slack_task, tg_task):
                if task is not None and not task.done():
                    task.cancel()
                    # swallow cancellation errors
                    with suppress(asyncio.CancelledError):
                        await task

    # Create app with lifespan
    app = FastAPI(
        title="AetherGraph Sidecar",
        version="0.1",
        lifespan=lifespan,
    )

    frontend_dir = Path(__file__).parent / "ui_static"
    if frontend_dir.exists():
        logger.info(f"Serving built frontend UI from {frontend_dir}")
        logger.info("UI will be available at: http://<host>:<port>/ui")

        # 1) Serve built assets under /ui/assets
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount(
                "/ui/assets",
                StaticFiles(directory=str(assets_dir)),
                name="ui_assets",
            )

        index_path = frontend_dir / "index.html"

        # 2) SPA catch-all: /ui and ANY /ui/... path -> index.html
        @app.get("/ui", include_in_schema=False)
        @app.get("/ui/{full_path:path}", include_in_schema=False)
        async def serve_ui(full_path: str = ""):
            if index_path.exists():
                return FileResponse(index_path)
            return PlainTextResponse(
                "UI bundle not found. Please build the frontend and copy it to ui_static.",
                status_code=501,
            )

    else:
        logger.warning(
            "AetherGraph UI bundle NOT found at %s. "
            "The /ui endpoint will return a 501 until you build and copy it.",
            frontend_dir,
        )

        @app.get("/ui", include_in_schema=False)
        async def ui_not_built():
            return PlainTextResponse(
                "UI bundle not found. Please build the frontend and copy it to ui_static.",
                status_code=501,
            )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # dev UI origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(router=runs_router, prefix="/api/v1")
    app.include_router(router=graphs_router, prefix="/api/v1")
    app.include_router(router=artifacts_router, prefix="/api/v1")
    app.include_router(router=memory_router, prefix="/api/v1")
    app.include_router(router=stats_router, prefix="/api/v1")
    app.include_router(router=identity_router, prefix="/api/v1")
    app.include_router(router=misc_router, prefix="/api/v1")
    app.include_router(router=vis_router, prefix="/api/v1")
    app.include_router(router=session_router, prefix="/api/v1")
    app.include_router(router=apps_router, prefix="/api/v1")
    app.include_router(router=agents_router, prefix="/api/v1")

    # Webui router
    from aethergraph.plugins.channel.routes.webui_routes import router as webui_router

    app.include_router(router=webui_router, prefix="/api/v1")

    # Install services globally so run()/tools see the same container
    install_services(container)

    # Optional: keep these for immediate access before lifespan runs
    app.state.settings = settings
    app.state.container = container

    return app


def _load_user_graphs_from_env() -> None:
    """
    Called inside each uvicorn worker to import user graphs based
    on environment variables set by the CLI.
    """
    modules_str = os.environ.get("AETHERGRAPH_LOAD_MODULES", "")
    paths_str = os.environ.get("AETHERGRAPH_LOAD_PATHS", "")
    project_root_str = os.environ.get("AETHERGRAPH_PROJECT_ROOT", ".")
    strict_str = os.environ.get("AETHERGRAPH_STRICT_LOAD", "0")

    modules = [m for m in modules_str.split(",") if m]
    paths = [Path(p) for p in paths_str.split(os.pathsep) if p]

    project_root = Path(project_root_str).resolve()
    strict = strict_str.lower() in ("1", "true", "yes")

    spec = LoadSpec(
        modules=modules,
        paths=paths,
        project_root=project_root,
        strict=strict,
    )

    loader = GraphLoader()
    report = loader.load(spec)

    # Optional: log report.loaded / report.errors here if you like
    print("🚀 [worker] Loaded user graphs:", report.loaded)
    if report.errors:
        for e in report.errors:
            print(f"⚠️ [worker load error] {e.source}: {e.error}")


def create_app_from_env() -> FastAPI:
    """
    Factory for uvicorn --reload / workers mode.
    Reads workspace + graph load config from env, imports user graphs,
    then builds the FastAPI app.
    """
    workspace = os.environ.get("AETHERGRAPH_WORKSPACE", "./aethergraph_data")
    log_level = os.environ.get("AETHERGRAPH_LOG_LEVEL", "warning")

    # 1) Load user graphs in *this* process
    _load_user_graphs_from_env()

    # 2) Build the app (your existing factory)
    # If you have a config system, wire it here
    app = create_app(
        workspace=workspace,
        cfg=None,  # or AppSettings.from_env(), etc.
        log_level=log_level,
    )
    return app
