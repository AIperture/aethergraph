import os

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.services.continuations.continuation import Correlator
from aethergraph.utils.optdeps import require


class SlackChannelAdapter(ChannelAdapter):
    capabilities: set[str] = {"text", "buttons", "image", "file", "edit", "stream"}

    def __init__(self, bot_token: str | None = None):
        """Slack channel adapter for handling Slack events.
        The bot token can be provided via the `SLACK_BOT_TOKEN` environment variable.
        The channel key format is: "slack:team/T:chan/C[:thread/TS]"
        """

        require(pkg="slack_sdk", extra="slack")
        from slack_sdk.web.async_client import AsyncWebClient

        self.client = AsyncWebClient(token=bot_token or os.environ["SLACK_BOT_TOKEN"])
        self._first_ts_by_chan: dict[str, str] = {}  # cache of first message ts by channel

    def _render_bar(self, percent: float, width: int = 20) -> str:
        p = max(0.0, min(1.0, float(percent)))
        filled = int(round(p * width))
        return "█" * filled + "░" * (width - filled)

    def _fmt_eta(self, sec) -> str:
        if sec is None:
            return ""
        try:
            s = int(max(0, float(sec)))
        except Exception:
            return ""
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

    @staticmethod
    def _parse(channel_key: str):
        """Parse the channel key into its components.
        E.g., "slack:team/T:chan/C[:thread/TS]" -> {"team": "T", "chan": "C", "thread": "TS"}
        """
        parts = channel_key.split(":")[1:]  # drop "slack"
        d = {}
        for p in parts:
            k, v = p.split("/", 1)
            d[k] = v
        return d

    async def _ensure_thread(self, channel_key: str, seed_text: str | None = None):
        meta = self._parse(channel_key)
        channel = meta["chan"]
        thread_ts = meta.get("thread")

        if thread_ts:
            return channel, thread_ts

        cached = self._first_ts_by_chan.get(channel_key)
        if cached:
            return channel, cached

        # Neutral root; DO NOT consume event.text
        resp = await self.client.chat_postMessage(channel=channel, text="(starting thread)")
        ts = resp.get("ts")
        self._first_ts_by_chan[channel_key] = ts
        return channel, ts

    async def peek_thread(self, channel_key: str):
        """
        Return the  thread_ts currently associated with the channel_key if know,
        without creating a new thread.
        """
        meta = self._parse(channel_key)
        if meta.get("thread"):
            return meta["thread"]
        # fallback to cache if first message ts (created ealier by _ensure_thread)
        return self._first_ts_by_chan.get(channel_key)

    async def send(self, event: OutEvent) -> dict | None:
        channel, thread_ts = await self._ensure_thread(event.channel)

        # streaming/upsert: we use chat.update keyed by upsert_key
        if (
            event.type
            in (
                "agent.stream.start",
                "agent.stream.delta",
                "agent.stream.end",
                "agent.message.update",
            )
            and event.upsert_key
        ):
            # stash ts per upsert_key inside thread cache
            key = (event.channel, event.upsert_key)
            ts = self._first_ts_by_chan.get(key)
            if ts is None:
                resp = await self.client.chat_postMessage(
                    channel=channel, thread_ts=thread_ts, text=event.text or "…"
                )
                ts = resp.get("ts")
                self._first_ts_by_chan[key] = ts
            else:
                if event.text:
                    # In slack, chat.update requires non-empty text for stream updates
                    await self.client.chat_update(channel=channel, ts=ts, text=event.text)
            return

        if event.type in ("session.need_approval", "link.buttons"):
            # Collect up to 5 buttons (Slack max per "actions" block)
            elements = []
            buttons = getattr(event, "buttons", None) or []
            if not buttons:
                # fallback to meta options
                opts = (event.meta or {}).get("options", ["Approve", "Reject"])
                buttons = [
                    # mimic button objects;
                    type(
                        "B",
                        (),
                        {"label": opts[0], "value": "approve", "style": "primary", "url": None},
                    ),
                    type(
                        "B",
                        (),
                        {"label": opts[-1], "value": "reject", "style": "danger", "url": None},
                    ),
                ]

            if len(buttons) > 5:
                self._warn("Slack supports max 5 buttons; truncating.")
                buttons = buttons[:5]

            for i, b in enumerate(buttons[:5]):  # Slack: max 5 elements
                btn: dict = {
                    "type": "button",
                    "text": {"type": "plain_text", "text": b.label, "emoji": True},
                }

                # Interactive buttons need an action_id; make them unique-ish
                btn["action_id"] = f"ag_button_{i}"

                # Either a URL button OR a value payload (not both)
                if getattr(b, "url", None):
                    btn["url"] = b.url
                else:
                    # pack choice + correlators into value for /slack/interact
                    value_payload = {
                        "choice": getattr(b, "value", None) or b.label,
                    }
                    # if passing correlators via event.meta
                    if event.meta:
                        for k in ("run_id", "node_id", "token"):
                            if k in event.meta:
                                value_payload[k] = event.meta[k]
                    import json

                    btn["value"] = json.dumps(value_payload)

                # Style: only set if valid
                style = getattr(b, "style", None)
                if style in ("primary", "danger"):
                    btn["style"] = style
                # else omit (default appearance)

                elements.append(btn)

            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": event.text or "Please approve:"},
                },
                {"type": "actions", "elements": elements},
            ]

            resp = await self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=event.text or "Please approve:",
                blocks=blocks,
            )
            return {
                "correlator": Correlator(
                    scheme="slack",
                    channel=event.channel,
                    thread=thread_ts,
                    message=resp.get("ts"),  # message ts
                )
            }

        # file upload (url or bytes)
        if event.type == "file.upload" and event.file:
            if "bytes" in event.file:
                await self.client.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    filename=event.file.get("filename", "file.bin"),
                    initial_comment=event.text,
                    file=event.file["bytes"],
                )
                return
            if "url" in event.file:
                # fall back to posting a link
                await self.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"{event.text or 'File'}: {event.file['url']}",
                )
                return

        # progress upsert (single message updated by upsert_key)
        if (
            event.type in ("agent.progress.start", "agent.progress.update", "agent.progress.end")
            and event.upsert_key
        ):
            r = event.rich or {}
            title = r.get("title") or "Working..."
            subtitle = r.get("subtitle") or ""
            total = r.get("total")
            current = r.get("current") or 0
            eta_seconds = r.get("eta_seconds")

            # compute percent + bar
            p = max(0.0, min(1.0, float(current) / float(total))) if total else 0.0
            bar = self._render_bar(p, 20) if total else ""
            pct_text = f"{int(round(p * 100))}%" if total else ""
            eta_text = self._fmt_eta(eta_seconds)
            header = f"⏳ {title}"
            if event.type == "agent.progress.end":
                header = f"{'✅' if (r.get('success', True)) else '⚠️'} {title}"
                if total:
                    bar = self._render_bar(1.0, 20)
                    pct_text = "100%"

            # Build Slack blocks
            blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{header}*"}}]
            if total:
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"`{bar}`  {pct_text}"}}
                )
            # optional subtitle + ETA
            ctx_tail = " • ".join(
                [t for t in (subtitle, f"ETA {eta_text}" if eta_text else "") if t]
            )
            if ctx_tail:
                blocks.append(
                    {"type": "context", "elements": [{"type": "mrkdwn", "text": ctx_tail}]}
                )

            # Upsert using the same cache dict already in use (keyed by (channel, upsert_key))
            key = (event.channel, event.upsert_key)
            ts = self._first_ts_by_chan.get(key)
            if ts is None:
                resp = await self.client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"{title} {pct_text}".strip(),
                    blocks=blocks,
                )
                self._first_ts_by_chan[key] = resp.get("ts")
            else:
                await self.client.chat_update(
                    channel=channel,
                    ts=ts,
                    text=f"{title} {pct_text}".strip() or "…",
                    blocks=blocks,
                )
            return

        # default: plain message, include (session.need_input) etc.
        resp = await self.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=event.text or ""
        )
        return {
            "correlator": Correlator(
                scheme="slack",
                channel=event.channel,
                thread=thread_ts,
                message=resp.get("ts"),  # message ts
            )
        }
