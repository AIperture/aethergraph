import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from aethergraph.api.v1.router import router as api_v1_router

# include apis
from aethergraph.config.config import AppSettings
from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings

# register all skills in the builtin agent (this is optional but keeps them together for now)
from aethergraph.core.runtime.runtime_services import (
    install_services,
    register_skills_from_path,
)

# from aethergraph.plugins.agents.agnet_buider_agent import *  # noqa: F403
# import built-in agents and plugins to register them
from aethergraph.plugins.agents.chat_agent.default_chat_agent import *  # noqa: F403

# from aethergraph.plugins.agents.graph_builder.agent import *  # noqa: F403
# from aethergraph.plugins.agents.aether_agent import *  # noqa: F403
# from aethergraph.plugins.agents.default_chat_agent_v2 import *  # noqa: F403
# channel routes
from aethergraph.server.loading import GraphLoader, LoadSpec
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.triggers.engine import TriggerEngine
from aethergraph.utils.optdeps import require

builtin_agent_skills_path = (
    Path(__file__).parent.parent / "plugins" / "agents" / "graph_builder" / "skills"
)


logger = logging.getLogger(__name__)


def create_app(
    *,
    workspace: str = "./aethergraph_workspace",
    cfg: Optional["AppSettings"] = None,
    log_level: str = "info",
) -> FastAPI:
    """
    Builds the FastAPI app, registers routers, and installs all services
    into app.state.container (and globally via install_services()).
    """

    # Resolve settings and container up front so lifespan can capture them
    settings = cfg or AppSettings()
    if settings.logging.console_level is None:
        settings.logging.console_level = log_level

    container = build_default_container(root=workspace, cfg=settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- Startup: attach settings/container and start external transports ---
        app.state.settings = settings
        app.state.container = container

        trigger_engine_task = None

        # Start trigger engine if trigger_service is present
        if hasattr(container, "trigger_engine") and container.trigger_engine is not None:
            trigger_engine: TriggerEngine = container.trigger_engine
            trigger_engine_task = asyncio.create_task(trigger_engine.run_forever())
            app.state.trigger_engine_task = trigger_engine_task
            logger.info("TriggerEngine background task started")

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

        # Register skills from the builtin path (optional, but keeps them together for now)
        logger.info(f"Registering skills from {builtin_agent_skills_path} for builtin agent...")
        register_skills_from_path(builtin_agent_skills_path, overwrite=True)

        # Replay persisted source registrations (tenant/global manifests).
        replay_strict = os.environ.get("AETHERGRAPH_REGISTRY_REPLAY_STRICT", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        replay_report = await container.registration_service.replay_registered_sources(
            strict=replay_strict
        )
        logger.info(
            "Registry replay complete: total=%s loaded=%s failed=%s",
            replay_report.total,
            replay_report.loaded,
            replay_report.failed,
        )
        if replay_report.errors:
            for err in replay_report.errors:
                logger.warning("Registry replay error: %s", err)
        try:
            # Hand control back to FastAPI / TestClient
            yield
        finally:
            # --- Shutdown: best-effort cleanup of background tasks ---
            # 1) Stop TriggerEngine gracefully
            if trigger_engine_task is not None:
                trigger_engine: TriggerEngine = container.trigger_engine
                try:
                    await trigger_engine.stop()
                except Exception:
                    logger.exception("Error stopping TriggerEngine")

                if not trigger_engine_task.done():
                    # In case it's still waiting on the poll sleep
                    trigger_engine_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await trigger_engine_task

            # 2) Stop Slack / Telegram tasks
            for task in (slack_task, tg_task):
                if task is not None and not task.done():
                    task.cancel()
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
        allow_origins=["http://localhost:5173", "null"],  # dev UI + file:// admin page
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(router=api_v1_router, prefix="/api/v1")

    # Conditionally load demo admin routes (not part of OSS core)
    _demo_svc_dir = (
        str(Path(settings.demo_service_dir).resolve()) if settings.demo_service_dir else None
    )
    if _demo_svc_dir and Path(_demo_svc_dir).is_dir():
        sys.path.insert(0, _demo_svc_dir)
        try:
            from admin_routes import router as demo_admin_router  # type: ignore[import-not-found]

            app.include_router(demo_admin_router, prefix="/api/v1")
            logger.info("Demo admin routes loaded from %s", _demo_svc_dir)
        except ImportError:
            logger.warning(
                "AETHERGRAPH_DEMO_SERVICE_DIR set but admin_routes not found in %s",
                _demo_svc_dir,
            )
        finally:
            sys.path.pop(0)

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

    # Permanently add project_root to sys.path so hot-loaded files
    # (e.g. via /api/v1/registry/register) can resolve local imports.
    # TODO(cloud): Same as __main__.py — skip this in cloud mode and use
    # per-request project_root in RegistrationService._register_source instead.
    pr_str = str(project_root)
    if pr_str not in sys.path:
        sys.path.insert(0, pr_str)

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
            if e.traceback:
                print(e.traceback)


def create_app_from_env() -> FastAPI:
    """
    Factory for uvicorn --reload / workers mode.
    Reads workspace + graph load config from env, imports user graphs,
    then builds the FastAPI app.
    """
    workspace = os.environ.get("AETHERGRAPH_WORKSPACE", "./aethergraph_workspace")
    log_level = os.environ.get("AETHERGRAPH_LOG_LEVEL", "warning")

    # 0) Load settings from env like `start_server` and CLI would (__main__.py)
    cfg = load_settings()
    set_current_settings(cfg)

    # 1) Load user graphs in *this* process
    _load_user_graphs_from_env()

    # 2) Build the app (your existing factory)
    # If you have a config system, wire it here
    app = create_app(
        workspace=workspace,
        cfg=cfg,
        log_level=log_level,
    )
    return app
