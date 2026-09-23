# channels/factory.py
from __future__ import annotations

import os
from typing import Any

from aethergraph.config.config import AppSettings
from aethergraph.plugins.channel.adapters.console import ConsoleChannelAdapter
from aethergraph.plugins.channel.adapters.file import FileChannelAdapter
from aethergraph.plugins.channel.adapters.slack import SlackChannelAdapter
from aethergraph.plugins.channel.adapters.telegram import TelegramChannelAdapter
from aethergraph.plugins.channel.adapters.webhook import WebhookChannelAdapter


def make_channel_adapters_from_env(cfg: AppSettings) -> dict[str, Any]:
    """Build delivery adapters from the active host settings.

    Examples:
        Build the default local adapters:
        ```python
        adapters = make_channel_adapters_from_env(AppSettings())
        ```

        Enable Telegram delivery:
        ```python
        settings = AppSettings()
        settings.telegram.enabled = True
        settings.telegram.bot_token = SecretStr("token")
        adapters = make_channel_adapters_from_env(settings)
        ```

    Args:
        cfg: Active host settings used to select and configure adapters.

    Returns:
        Delivery adapters keyed by canonical channel prefix.

    Notes:
        Provider transports receive messages separately. These adapters only
        deliver outbound channel messages.
    """
    # Always include console adapter
    adapters = {"console": ConsoleChannelAdapter()}

    # include Slack adapter if enabled
    if cfg.slack.enabled and cfg.slack.bot_token:
        adapters["slack"] = SlackChannelAdapter(bot_token=cfg.slack.bot_token.get_secret_value())

    # include Telegram adapter if enabled
    if cfg.telegram.enabled and cfg.telegram.bot_token:
        adapters["tg"] = TelegramChannelAdapter(bot_token=cfg.telegram.bot_token.get_secret_value())

    # include default file adapter
    file_root = os.path.join(cfg.workspace, "channel_files")
    adapters["file"] = FileChannelAdapter(root=file_root)

    # include webhook adapter
    adapters["webhook"] = WebhookChannelAdapter()

    return adapters
