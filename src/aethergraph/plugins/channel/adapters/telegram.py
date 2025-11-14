import asyncio
import logging
import os

import httpx

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.services.continuations.continuation import Correlator


def _tg_render_bar(percent: float, width: int = 20) -> str:
    p = max(0.0, min(1.0, percent))
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def _tg_fmt_eta(sec: float | None) -> str:
    if sec is None:
        return ""
    s = int(max(0, sec))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _prune(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _mk_params(chat_id: int, topic_id: int | None, **rest) -> dict:
    p = {"chat_id": chat_id, **rest}
    if topic_id is not None:
        p["message_thread_id"] = topic_id
    return p


def _safe_text_md(text: str | None) -> tuple[str, str | None]:
    """
    Best-effort: if text looks like Markdown-safe, return ("Markdown", text).
    Else, drop parse mode to avoid 400s on unescaped symbols.
    """
    if not text:
        return "", "Markdown"
    # very light check – if it contains risky characters unbalanced, avoid MD
    risky = any(c in text for c in ("*", "_", "[", "`"))
    return (text, None if risky else "Markdown")


class TelegramChannelAdapter(ChannelAdapter):
    """
    Telegram channel adapter using the Bot API.
    Channel key format:
      - "tg:chat/<chat_id>"
      - Optional topic (supergroups): "tg:chat/<chat_id>:topic/<message_thread_id>"
    """

    capabilities: set[str] = {"text", "buttons", "image", "file", "edit", "stream"}

    def __init__(self, bot_token: str | None = None, *, timeout_s: int = 15):
        self.token = bot_token or os.environ["TELEGRAM_BOT_TOKEN"]
        self.base = f"https://api.telegram.org/bot{self.token}"

        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        limits = httpx.Limits(
            max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0
        )

        try:
            transport = httpx.AsyncHTTPTransport(retries=0, local_address="0.0.0.0", http2=False)
        except Exception:
            transport = httpx.AsyncHTTPTransport(retries=0, http2=False)

        # proxies = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None

        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, transport=transport)
        # cache for edit/upsert: (channel_key, upsert_key) -> (chat_id, message_id)
        self._msg_id_cache: dict[tuple[str, str], tuple[int, int]] = {}

    async def aclose(self):
        try:
            await self._client.aclose()
        except Exception as e:
            logger = logging.getLogger("aethergraph.plugins.channel.adapters.telegram")
            logger.warning(f"Failed to close Telegram client: {e}")

    # ------------- helpers -------------
    @staticmethod
    def _parse(channel_key: str) -> dict:
        """
        Parse "tg:chat/<chat_id>[:topic/<message_thread_id>]" → {"chat": int, "topic": int|None}
        """
        if not channel_key.startswith("tg:"):
            raise ValueError(f"Not a telegram channel key: {channel_key}")
        parts = channel_key.split(":")[1:]  # drop "tg"
        d = {}
        for p in parts:
            k, v = p.split("/", 1)
            d[k] = v
        chat_id = int(d["chat"])
        topic_id = int(d["topic"]) if "topic" in d else None
        return {"chat": chat_id, "topic": topic_id}

    async def _api(self, method: str, **params):
        """POST to Telegram with retries on connect and 429, robust error handling."""
        url = f"{self.base}/{method}"
        files = params.pop("_files", None)

        last_exc = None
        for attempt in range(3):
            try:
                if files:
                    resp = await self._client.post(url, data=_prune(params), files=files)
                else:
                    resp = await self._client.post(url, json=_prune(params))
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok", False):
                    if data.get("error_code") == 429:
                        retry_after = (data.get("parameters") or {}).get("retry_after", 1)
                        await asyncio.sleep(int(retry_after))
                        continue
                    desc = data.get("description", "Unknown Telegram error")
                    raise RuntimeError(f"Telegram API error: {data.get('error_code')} {desc}")
                return data
            except httpx.ConnectError as e:
                last_exc = e
                await asyncio.sleep(0.6 * (attempt + 1))
            except httpx.ReadTimeout as e:
                last_exc = e
                await asyncio.sleep(0.6 * (attempt + 1))
            except ValueError as e:
                text = getattr(resp, "text", lambda: "")()
                raise RuntimeError(f"Telegram non-JSON response: {text[:200]}") from e

        raise httpx.ConnectError(
            f"Failed to call Telegram {method}; last_error={last_exc!r}"
        ) from last_exc

    # ------------- core send -------------
    async def peek_thread(self, channel_key: str) -> str | None:
        meta = self._parse(channel_key)
        return str(meta["topic"]) if meta["topic"] is not None else ""

    async def send(self, event: OutEvent) -> dict | None:
        meta = self._parse(event.channel)
        chat_id = meta["chat"]
        topic_id = meta["topic"]  # None if not provided

        # Streaming & upsert (editMessageText)
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
            key = (event.channel, event.upsert_key)
            if key not in self._msg_id_cache:
                text, md = _safe_text_md(event.text or "…")
                params = _mk_params(chat_id, topic_id, text=text, parse_mode=md)
                resp = await self._api("sendMessage", **params)
                msg = resp["result"]
                self._msg_id_cache[key] = (msg["chat"]["id"], msg["message_id"])
            else:
                ch, mid = self._msg_id_cache[key]
                if event.text:
                    text, md = _safe_text_md(event.text)
                    await self._api(
                        "editMessageText", chat_id=ch, message_id=mid, text=text, parse_mode=md
                    )
            return None

        # Buttons / approvals
        if event.type in ("session.need_approval", "link.buttons"):
            buttons = getattr(event, "buttons", None) or []
            if not buttons:
                opts = (event.meta or {}).get("options", ["Approve", "Reject"])
                buttons = [
                    type(
                        "B", (), {"label": opts[0], "value": "approve", "style": None, "url": None}
                    ),
                    type(
                        "B", (), {"label": opts[-1], "value": "reject", "style": None, "url": None}
                    ),
                ]

            # Compact callback data: "c=<choice>|k=<resume_key>"  (<< 64 bytes)
            resume_key = (event.meta or {}).get("resume_key") or ""
            rows = []
            for b in buttons[:8]:
                label = b.label
                val = getattr(b, "value", None) or label
                if getattr(b, "url", None):
                    rows.append([{"text": label, "url": b.url}])
                else:
                    data = f"c={str(val)[:20]}|k={resume_key}"
                    rows.append([{"text": label, "callback_data": data}])

            reply_markup = {"inline_keyboard": rows}
            text, md = _safe_text_md(event.text or "Please approve:")

            params = _mk_params(
                chat_id, topic_id, text=text, parse_mode=md, reply_markup=reply_markup
            )
            resp = await self._api("sendMessage", **params)
            msg = resp["result"]

            return {
                "correlator": Correlator(
                    scheme="tg",
                    channel=event.channel,
                    thread=str(topic_id or ""),
                    message=str(msg["message_id"]),
                )
            }

        # File upload
        if event.type == "file.upload" and event.file:
            filename = event.file.get("filename", "file.bin")
            caption = event.text or filename
            if "bytes" in event.file:
                files = {"document": (filename, event.file["bytes"])}
                params = _mk_params(chat_id, topic_id, caption=caption)
                await self._api("sendDocument", _files=files, **params)
                return None
            if "url" in event.file:
                text, md = _safe_text_md(f"{caption}: {event.file['url']}")
                params = _mk_params(chat_id, topic_id, text=text, parse_mode=md)
                await self._api("sendMessage", **params)
                return None

        # Progress with upsert/edit (single text body)
        if (
            event.type in ("agent.progress.start", "agent.progress.update", "agent.progress.end")
            and event.upsert_key
        ):
            r = event.rich or {}
            title = r.get("title") or "Working..."
            subtitle = r.get("subtitle") or ""
            total = r.get("total")
            cur = r.get("current") or 0
            pct = max(0.0, min(1.0, float(cur) / float(total))) if total else 0.0
            bar = _tg_render_bar(pct, 20)
            pct_txt = f"{int(round(pct * 100))}%"
            eta_txt = _tg_fmt_eta(r.get("eta_seconds"))
            header = f"⏳ {title}"
            if event.type == "agent.progress.end":
                header = f"{'✅' if r.get('success', True) else '⚠️'} {title}"
                if total:
                    bar = _tg_render_bar(1.0, 20)
                    pct_txt = "100%"

            body_lines = [f"*{header}*"]
            if total:
                body_lines.append(f"`{bar}`  {pct_txt}")
            tail = " • ".join([t for t in (subtitle, f"ETA {eta_txt}" if eta_txt else "") if t])
            if tail:
                body_lines.append(tail)
            text = "\n".join(body_lines)

            key = (event.channel, event.upsert_key)
            if key not in self._msg_id_cache:
                t, md = _safe_text_md(text)
                params = _mk_params(chat_id, topic_id, text=t, parse_mode=md)
                resp = await self._api("sendMessage", **params)
                msg = resp["result"]
                self._msg_id_cache[key] = (msg["chat"]["id"], msg["message_id"])
            else:
                ch, mid = self._msg_id_cache[key]
                t, md = _safe_text_md(text)
                await self._api(
                    "editMessageText", chat_id=ch, message_id=mid, text=t, parse_mode=md
                )
            return None

        # Image (sendPhoto)
        if getattr(event, "image", None):
            url = event.image.get("url", "")
            caption = event.text or event.image.get("title") or ""
            params = _mk_params(chat_id, topic_id, photo=url, caption=caption)
            await self._api("sendPhoto", **params)
            return None

        # Default: plain message
        t, md = _safe_text_md(event.text or "")
        params = _mk_params(chat_id, topic_id, text=t, parse_mode=md)
        resp = await self._api("sendMessage", **params)
        return {
            "correlator": Correlator(
                scheme="tg",
                channel=event.channel,
                thread=str(topic_id or ""),
                message=str(resp["result"]["message_id"]),
            )
        }
