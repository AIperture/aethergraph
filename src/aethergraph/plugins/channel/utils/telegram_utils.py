import hmac
import inspect
import json
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException, Request

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.plugins.channel.utils.turn_dispatch import (
    attachments_from_incoming_files,
    dispatch_channel_turn_run,
)
from aethergraph.services.channel.ingress import (
    ChannelIngress,
    IncomingFile,
    IncomingMessage,
)
from aethergraph.services.continuations.continuation import Correlator

router = APIRouter()

# Reuse one aiohttp session with timeouts
_aiohttp_session: aiohttp.ClientSession | None = None


def _http_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(
            total=40,  # > 30
            connect=5,
            sock_read=35,  # > 30
        )
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _aiohttp_session


def _verify_secret(request: Request):
    TELEGRAM_WEBHOOK_SECRET = (
        request.app.state.settings.telegram.webhook_secret.get_secret_value() or ""
    )
    if not TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(401, "no telegram webhook secret configured")
    if TELEGRAM_WEBHOOK_SECRET:
        hdr = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not hmac.compare_digest(hdr or "", TELEGRAM_WEBHOOK_SECRET):
            raise HTTPException(401, "bad telegram webhook secret")


def _channel_key(chat_id: int, topic_id: int | None) -> str:
    base = f"tg:chat/{int(chat_id)}"
    return f"{base}:topic/{int(topic_id)}" if topic_id else base


def _tg_scheme_and_channel_id(chat_id: int, topic_id: int | None) -> tuple[str, str]:
    """
    Map Telegram chat/topic to (scheme, channel_id) pair for ChannelIngress.

    _channel_key(chat_id, topic_id) builds:
      "tg:chat/<id>" or "tg:chat/<id>:topic/<topic_id>"

    So we use:
      scheme    = "tg"
      channel_id = "chat/<id>" or "chat/<id>:topic/<topic_id>"
    """
    base = f"chat/{int(chat_id)}"
    if topic_id:
        base = f"{base}:topic/{int(topic_id)}"
    return "tg", base


# ---- helpers ----
async def _tg_get_file_path(file_id: str, token: str) -> str | None:
    if not token:
        return None
    api = f"https://api.telegram.org/bot{token}/getFile"
    async with _http_session().post(api, json={"file_id": file_id}) as r:
        if r.status != 200:
            return None
        data = await r.json()
        if not data.get("ok"):
            return None
        return (data.get("result") or {}).get("file_path")


async def _tg_download_file(file_path: str, token: str) -> bytes:
    url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    async with _http_session().get(url) as r:
        r.raise_for_status()
        return await r.read()


async def _process_update(container, payload: dict, token: str):
    ingress: ChannelIngress = container.channel_ingress

    try:
        # 1) Callback queries (inline buttons) -------------------------
        cq = payload.get("callback_query")
        if cq:
            msg = cq.get("message") or {}
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            topic_id = msg.get("message_thread_id")
            ch_key = _channel_key(chat_id, topic_id)

            data_raw = cq.get("data") or ""
            choice = "reject"
            resume_key = None

            # Accept JSON or compact "c=...|k=..." forms
            try:
                data = json.loads(data_raw)
                choice = str(data.get("choice", "reject"))
                resume_key = data.get("resume_key") or data.get("k")
            except Exception:
                try:
                    parts = dict(p.split("=", 1) for p in data_raw.split("|") if "=" in p)
                    choice = parts.get("c", parts.get("i", "reject"))
                    resume_key = parts.get("k")
                except Exception:
                    choice = str(data_raw)

            tok = None
            run_id = None
            node_id = None

            # Preferred: resolve alias → token
            if resume_key and hasattr(container.cont_store, "token_from_alias"):
                maybe_tok = container.cont_store.token_from_alias(resume_key)
                tok = await maybe_tok if inspect.isawaitable(maybe_tok) else maybe_tok

            if tok and hasattr(container.cont_store, "get_by_token"):
                maybe_cont = container.cont_store.get_by_token(tok)
                cont = await maybe_cont if inspect.isawaitable(maybe_cont) else maybe_cont
                if cont:
                    run_id, node_id = cont.run_id, cont.node_id

            # Fallback: thread-scoped correlator
            if not tok:
                corr = Correlator(
                    scheme="tg",
                    channel=ch_key,
                    thread=str(topic_id or ""),
                    message="",
                )
                cont = await container.cont_store.find_by_correlator(corr=corr)
                if cont:
                    run_id, node_id, tok = cont.run_id, cont.node_id, cont.token

            resumed = await ingress.handle(
                IncomingMessage(
                    scheme="tg",
                    channel_id=f"chat/{int(chat_id)}"
                    + (f":topic/{int(topic_id)}" if topic_id is not None else ""),
                    thread_id=str(topic_id or ""),
                    text="",
                    choice=choice,
                    conversation_id=f"tg:chat/{int(chat_id)}"
                    + (f":topic/{int(topic_id)}" if topic_id is not None else ""),
                    meta={
                        "telegram": {
                            "callback_id": cq.get("id"),
                            "message_id": msg.get("message_id"),
                            "chat_id": chat_id,
                        },
                    },
                )
            )

            if (not resumed) and tok and run_id and node_id:
                await container.resume_router.resume(
                    run_id=run_id,
                    node_id=node_id,
                    token=tok,
                    payload={
                        "choice": choice,
                        "matched": True,
                        "text": "",
                        "telegram": {
                            "callback_id": cq.get("id"),
                            "message_id": msg.get("message_id"),
                            "chat_id": chat_id,
                        },
                    },
                )

            # Ack the button press to stop the spinner
            try:
                tg_adapter = container.channels.adapters.get("tg")
                if tg_adapter:
                    await tg_adapter._api("answerCallbackQuery", callback_query_id=cq.get("id"))
            except Exception:
                pass
            return

        # 2) Regular messages / uploads -------------------------------
        msg = payload.get("message")
        if not msg:
            return
        if (msg.get("from") or {}).get("is_bot"):
            return

        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        topic_id = msg.get("message_thread_id")
        ch_key = _channel_key(chat_id, topic_id)
        scheme, channel_id = _tg_scheme_and_channel_id(chat_id, topic_id)

        text = (msg.get("text") or msg.get("caption") or "") or ""

        tg_files: list[dict[str, Any]] = []

        # Photos
        photos = msg.get("photo") or []
        if photos:
            ph = photos[-1]
            file_id = ph.get("file_id")
            size = ph.get("file_size")
            name = f"photo_{file_id}.jpg"
            file_path = await _tg_get_file_path(file_id, token)
            if file_path:
                try:
                    data = await _tg_download_file(file_path, token)
                    uri = await _stage_and_save(
                        container, data=data, name=name, ch_key=ch_key, cont=None
                    )
                    tg_files.append(
                        _file_ref(
                            file_id=file_id,
                            name=name,
                            mimetype="image/jpeg",
                            size=size,
                            uri=uri,
                            ch_key=ch_key,
                            ts=msg.get("date"),
                        )
                    )
                except Exception as e:
                    container.logger and container.logger.for_run().warning(
                        f"Telegram photo download failed: {e}"
                    )

        # Documents
        doc = msg.get("document")
        if doc:
            file_id = doc.get("file_id")
            size = doc.get("file_size")
            name = doc.get("file_name") or f"document_{file_id}"
            mime = _normalize_mime_by_name(name, doc.get("mime_type"))
            file_path = await _tg_get_file_path(file_id, token)
            if file_path:
                try:
                    data = await _tg_download_file(file_path, token)
                    uri = await _stage_and_save(
                        container, data=data, name=name, ch_key=ch_key, cont=None
                    )
                    tg_files.append(
                        _file_ref(
                            file_id=file_id,
                            name=name,
                            mimetype=mime,
                            size=size,
                            uri=uri,
                            ch_key=ch_key,
                            ts=msg.get("date"),
                        )
                    )
                except Exception as e:
                    container.logger and container.logger.for_run().warning(
                        f"Telegram document download failed: {e}"
                    )

        # Turn Telegram file_refs into IncomingFile with pre-saved URIs
        incoming_files: list[IncomingFile] = []
        for fr in tg_files:
            incoming_files.append(
                IncomingFile(
                    id=fr["id"],
                    name=fr["name"],
                    mimetype=fr.get("mimetype"),
                    size=fr.get("size"),
                    uri=fr.get("uri"),  # already staged as artifact
                    url=None,  # no re-download
                    extra={
                        "platform": "telegram",
                        "channel_key": fr.get("channel_key"),
                        "ts": fr.get("ts"),
                    },
                )
            )

        meta = {
            "raw": payload,
            "channel_key": ch_key,
            "telegram": {
                "message_id": msg.get("message_id"),
                "chat_id": chat_id,
            },
        }

        resumed = await ingress.handle(
            IncomingMessage(
                scheme=scheme,
                channel_id=channel_id,
                thread_id=str(topic_id or ""),
                text=text,
                files=incoming_files or None,
                conversation_id=f"tg:{channel_id}",
                meta=meta,
            )
        )

        if (not resumed) and (text or incoming_files):
            default_agent_id = getattr(getattr(container, "settings", None), "telegram", None)
            default_agent_id = getattr(default_agent_id, "default_agent_id", None)
            if default_agent_id:
                await dispatch_channel_turn_run(
                    container=container,
                    identity=RequestIdentity(user_id="local", org_id="local", mode="local"),
                    agent_id=default_agent_id,
                    text=text,
                    attachments=attachments_from_incoming_files(tg_files, source="telegram_upload"),
                    user_meta={
                        **meta,
                        "channel_key": ch_key,
                        "conversation_id": f"tg:{channel_id}",
                    },
                    tags=[
                        f"channel:{ch_key}",
                        f"conversation:tg:{channel_id}",
                        f"agent:{default_agent_id}",
                    ],
                )

        container.logger and container.logger.for_run().debug(
            f"[TG] inbound: text={text!r} files={len(incoming_files)} resumed={resumed}"
        )

    except Exception as e:
        container.logger and container.logger.for_run().error(
            f"Telegram inbound processing error: {e}", exc_info=True
        )


# ---- file helpers ----
def _normalize_mime_by_name(name: str | None, hint: str | None) -> str:
    extmap = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "csv": "text/csv",
        "json": "application/json",
        "yaml": "text/yaml",
        "yml": "text/yaml",
        "txt": "text/plain",
        "md": "text/markdown",
        "zip": "application/zip",
        "gz": "application/gzip",
        "tar": "application/x-tar",
        "7z": "application/x-7z-compressed",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "mph": "application/octet-stream",
    }
    if hint:
        return hint.lower()
    if name and "." in name:
        ext = name.lower().rsplit(".", 1)[-1]
        return extmap.get(ext, "application/octet-stream")
    return "application/octet-stream"


async def _stage_and_save(container, *, data: bytes, name: str, ch_key: str, cont) -> str:
    tmp = await container.artifacts.plan_staging_path(planned_ext=f"_{name}")

    with open(tmp, "wb") as f:
        f.write(data)
    run_id = cont.run_id if cont else "ad-hoc"
    node_id = cont.node_id if cont else "telegram"
    art = await container.artifacts.save_file(
        path=tmp,
        kind="upload",
        run_id=run_id,
        graph_id="channel",
        node_id=node_id,
        tool_name="telegram.upload",
        tool_version="0.0.1",
        labels={"source": "telegram", "channel": ch_key, "name": name},
    )
    return getattr(art, "uri", None) or f"file://{tmp}"


def _file_ref(
    *, file_id: str, name: str, mimetype: str, size: int | None, uri: str, ch_key: str, ts: Any
):
    return {
        "id": file_id,
        "name": name,
        "mimetype": mimetype or "",
        "size": size,
        "uri": uri,
        "url_private": None,
        "platform": "telegram",
        "channel_key": ch_key,
        "ts": ts,
    }
