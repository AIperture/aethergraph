import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aethergraph.api.v1.artifacts import router as artifacts_router
from aethergraph.api.v1.graphs import router as graphs_router
from aethergraph.api.v1.memory import router as memory_router
from aethergraph.api.v1.runs import router as runs_router
from aethergraph.api.v1.stats import router as stats_router

# include apis
from aethergraph.config.config import AppSettings
from aethergraph.core.runtime.runtime_services import install_services

# channel routes
from aethergraph.services.container.default_container import build_default_container
from aethergraph.utils.optdeps import require


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

    # Install services globally so run()/tools see the same container
    install_services(container)

    # Optional: keep these for immediate access before lifespan runs
    app.state.settings = settings
    app.state.container = container

    return app


def create_app_old(
    *,
    workspace: str = "./aethergraph_data",
    cfg: Optional["AppSettings"] = None,
    log_level: str = "info",
) -> FastAPI:
    """
    Builds the FastAPI app, registers routers, and installs all services
    into app.state.container (and globally via install_services()).
    """
    app = FastAPI(title="AetherGraph Sidecar", version="0.1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # dev UI origin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Resolve settings early, so we can conditionally include routers
    settings = cfg or AppSettings()
    app.state.settings = settings

    # --- API Routers ---
    app.include_router(router=runs_router, prefix="/api/v1")
    app.include_router(router=graphs_router, prefix="/api/v1")
    app.include_router(router=artifacts_router, prefix="/api/v1")
    app.include_router(router=memory_router, prefix="/api/v1")

    # --- Routers (HTTP transports) ---
    # For now, we can just always include; or gate it with a flag like settings.slack.use_webhook.
    # app.include_router(slack_router)     # HTTP /slack/events + /slack/interact
    # app.include_router(console_router)
    # app.include_router(telegram_router)
    # app.include_router(webui_router)

    # override log level in config
    settings.logging.level = log_level

    # ---- Services container ----
    container = build_default_container(root=workspace, cfg=settings)
    app.state.container = container

    # install globally so run()/tools see the same services
    install_services(container)

    # ---- External channel transports (Socket Mode, polling, etc.) ----
    @app.on_event("startup")
    async def start_external_transports():
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
            asyncio.create_task(runner.start())

        # Telegram polling for local / dev
        tg_cfg = settings.telegram
        if tg_cfg and tg_cfg.enabled and tg_cfg.polling_enabled and tg_cfg.bot_token:
            from ..plugins.channel.websockets.telegram_polling import TelegramPollingRunner

            tg_runner = TelegramPollingRunner(container=container, settings=settings)
            app.state.telegram_polling_runner = tg_runner
            asyncio.create_task(tg_runner.start())

    return app
