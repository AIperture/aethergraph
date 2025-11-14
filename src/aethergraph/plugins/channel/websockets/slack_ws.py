from aethergraph.utils.optdeps import require

from ..utils.slack_utils import handle_slack_events_common, handle_slack_interactive_common

try:
    require(pkg="slack_sdk", extra="slack")
    from slack_sdk.socket_mode.aiohttp import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    raise ImportError(
        "slack_sdk is required for SlackSocketModeRunner; please install aethergraph with the [slack] extra."
    ) from None


class SlackSocketModeRunner:
    def __init__(self, container, settings):
        self.container = container
        self.settings = settings

        self.bot_token = (
            settings.slack.bot_token.get_secret_value() if settings.slack.bot_token else ""
        )
        self.app_token = (
            settings.slack.app_token.get_secret_value() if settings.slack.app_token else ""
        )  # xapp-...

        self.web_client = AsyncWebClient(token=self.bot_token)
        self.client: SocketModeClient | None = None

    async def _handle_socket_request(self, client: SocketModeClient, req: SocketModeRequest):
        # events from Slack
        if req.type == "events_api":
            # req.payload has same shape as HTTP Events API body
            await handle_slack_events_common(self.container, self.settings, req.payload)
            await client.send_socket_mode_response({"envelope_id": req.envelope_id})
            return

        # interactive actions (buttons, etc.)
        if req.type == "interactive":
            # payload is already parsed JSON dict
            payload = req.payload
            await handle_slack_interactive_common(self.container, payload)
            await client.send_socket_mode_response({"envelope_id": req.envelope_id})
            return

        # other request types (slash commands, shortcuts, etc.) can be added later
        await client.send_socket_mode_response({"envelope_id": req.envelope_id})

    async def start(self):
        lg = self.container.logger.for_run()
        if not (self.bot_token and self.app_token):
            lg.warning(
                "[Slack SocketMode] bot_token or app_token not configured; skipping Socket Mode startup."
            )
            return

        self.client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.web_client,
        )
        # register listener
        self.client.socket_mode_request_listeners.append(self._handle_socket_request)

        lg.info("[Slack SocketMode] connecting to Slack...")
        await self.client.connect()
        # NOTE: this call returns immediately; the internal loop lives with the event loop
        lg.info("[Slack SocketMode] connected.")
