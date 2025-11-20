# channels/factory.py
import os
from typing import Any

from aethergraph.config.config import AppSettings
from aethergraph.plugins.channel.adapters.console import ConsoleChannelAdapter
from aethergraph.plugins.channel.adapters.file import FileChannelAdapter
from aethergraph.plugins.channel.adapters.slack import SlackChannelAdapter
from aethergraph.plugins.channel.adapters.telegram import TelegramChannelAdapter
from aethergraph.plugins.channel.adapters.webhook import WebhookChannelAdapter
from aethergraph.services.channel.channel_bus import ChannelBus


def make_channel_adapters_from_env(cfg: AppSettings) -> dict[str, Any]:
    print("🔌 Setting up channel adapters... with config:", cfg.telegram)
    # Always include console adapter
    adapters = {"console": ConsoleChannelAdapter()}

    # include Slack adapter if enabled
    if cfg.slack.enabled and cfg.slack.bot_token and cfg.slack.signing_secret:
        adapters["slack"] = SlackChannelAdapter(bot_token=cfg.slack.bot_token.get_secret_value())

    # include Telegram adapter if enabled
    if cfg.telegram.enabled and cfg.telegram.bot_token:
        adapters["tg"] = TelegramChannelAdapter(bot_token=cfg.telegram.bot_token.get_secret_value())

    # include default file adapter
    file_root = os.path.join(cfg.root, "channel_files")
    adapters["file"] = FileChannelAdapter(root=file_root)

    # include webhook adapter
    adapters["webhook"] = WebhookChannelAdapter()
    return adapters


def build_bus(
    adapters: dict[str, Any],
    default: str = "console:stdin",
    logger=None,
    resume_router=None,
    cont_store=None,
) -> ChannelBus:
    bus = ChannelBus(adapters, logger=logger, resume_router=resume_router, store=cont_store)
    bus.set_default_channel_key(default)
    return bus
