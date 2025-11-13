import asyncio
import os
import secrets
from typing import Optional
from aethergraph.config.config import AppSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..services.container.default_container import DefaultContainer, build_default_container 
from ..core.runtime.runtime_services import install_services 

# channel routes 
from ..plugins.channel.routes.console_routes import router as console_router
from ..plugins.channel.routes.telegram_routes import router as telegram_router
from ..plugins.channel.routes.slack_routes import router as slack_router 
from ..plugins.channel.routes.webui_routes import install_web_channel, webui_router
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
        if (
            tg_cfg
            and tg_cfg.enabled
            and tg_cfg.polling_enabled
            and tg_cfg.bot_token
        ):
            from ..plugins.channel.websockets.telegram_polling import TelegramPollingRunner
            tg_runner = TelegramPollingRunner(container=container, settings=settings)
            app.state.telegram_polling_runner = tg_runner
            asyncio.create_task(tg_runner.start())

    return app