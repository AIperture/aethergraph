import hashlib
import hmac
import json
import time

import aiohttp
from fastapi import HTTPException, Request

from aethergraph.services.channel.ingress import (
    ChannelIngress,
    IncomingFile,
    IncomingMessage,
)


# --- shared utils ---
async def _download_slack_file(url: str, token: str) -> bytes:
    async with (
        aiohttp.ClientSession() as sess,
        sess.get(url, headers={"Authorization": f"Bearer {token}"}) as r,
    ):
        r.raise_for_status()
        return await r.read()


def _slack_scheme_and_channel_id(team_id: str | None, channel_id: str | None) -> tuple[str, str]:
    """
    Map Slack team/channel to the (scheme, channel_id) pair used by ChannelIngress.

    We keep the existing Slack channel key shape:
      ch_key = "slack:team/T:chan/C"
    and split it as:
      scheme = "slack"
      channel_id = "team/T:chan/C"
    """
    team = team_id or "unknown"
    chan = channel_id or "unknown"
    # This matches your existing _channel_key base form.
    return "slack", f"team/{team}:chan/{chan}"


def _verify_sig(request: Request, body: bytes):
    """Verify Slack request signature (HTTP webhooks only)."""
    SLACK_SIGNING_SECRET = (
        request.app.state.settings.slack.signing_secret.get_secret_value()
        if request.app.state.settings.slack.signing_secret
        else ""
    )
    if not SLACK_SIGNING_SECRET:
        raise HTTPException(401, "no slack signing secret configured")

    ts = request.headers.get("X-Slack-Request-Timestamp")
    sig = request.headers.get("X-Slack-Signature")
    if not ts or not sig or abs(time.time() - int(ts)) > 300:
        raise HTTPException(400, "stale or missing signature")
    basestring = f"v0:{ts}:{body.decode()}"
    my_sig = (
        "v0="
        + hmac.new(SLACK_SIGNING_SECRET.encode(), basestring.encode(), hashlib.sha256).hexdigest()
    )
    if not hmac.compare_digest(my_sig, sig):
        raise HTTPException(401, "bad signature")


def _channel_key(team_id: str, channel_id: str, thread_ts: str | None) -> str:
    """Construct a Slack channel key from its components.
    E.g., team_id="T", channel_id="C", thread_ts="TS" -> "slack:team/T:chan/C:thread/TS"
    """
    key = f"slack:team/{team_id}:chan/{channel_id}"
    if thread_ts:
        key += f":thread/{thread_ts}"
    return key


async def _stage_and_save(c, *, data: bytes, file_id: str, name: str, ch_key: str, cont) -> str:
    """Write bytes to tmp path, then save via FileArtifactStore.save_file(...).
    Returns the Artifact.uri (string)."""
    tmp = c.artifacts.tmp_path(suffix=f"_{file_id}")
    with open(tmp, "wb") as f:
        f.write(data)
    run_id = cont.run_id if cont else "ad-hoc"
    node_id = cont.node_id if cont else "channel"
    # graph_id is unknown here; set a neutral tag
    art = await c.artifacts.save_file(
        path=tmp,
        kind="upload",
        run_id=run_id,
        graph_id="channel",
        node_id=node_id,
        tool_name="slack.upload",
        tool_version="0.0.1",
        suggested_uri=None,
        pin=False,
        labels={"source": "slack", "slack_file_id": file_id, "channel": ch_key, "name": name},
        metrics=None,
        preview_uri=None,
    )
    return getattr(art, "uri", None) or getattr(art, "path", None) or f"file://{tmp}"


async def handle_slack_events_common(container, settings, payload: dict) -> dict:
    """
    Common handler for Slack Events API payloads.
    Now delegates continuation lookup & resume to ChannelIngress.
    """
    SLACK_BOT_TOKEN = (
        settings.slack.bot_token.get_secret_value() if settings.slack.bot_token else ""
    )
    c = container
    ingress: ChannelIngress = c.channel_ingress  # must exist in your container

    ev = payload.get("event") or {}
    ev_type = ev.get("type")
    thread_ts = ev.get("thread_ts") or ev.get("ts")

    # --- message (user -> bot) ---
    if ev_type == "message" and not ev.get("bot_id"):
        team = payload.get("team_id")
        chan = ev.get("channel")
        text = ev.get("text", "") or ""
        files = ev.get("files") or []

        # Full Slack key for labels/metadata
        ch_key = _channel_key(team, chan, None)  # "slack:team/T:chan/C"
        scheme, channel_id = _slack_scheme_and_channel_id(team, chan)  # ("slack", "team/T:chan/C")

        # --- Slack-specific file download + artifact save ---
        file_refs: list[dict] = []
        if files:
            token = SLACK_BOT_TOKEN
            for f in files:
                if f.get("mode") == "tombstone":
                    continue
                file_id = f.get("id")
                name = f.get("name") or f.get("title") or "file"
                mimetype = f.get("mimetype")
                size = f.get("size")
                url_priv = f.get("url_private") or f.get("url_private_download")

                uri = None
                if url_priv and token:
                    try:
                        data_bytes = await _download_slack_file(url_priv, token)
                        # use Slack-specific labels via _stage_and_save
                        uri = await _stage_and_save(
                            c,
                            data=data_bytes,
                            file_id=file_id,
                            name=name,
                            ch_key=ch_key,
                            cont=None,  # we don't know cont yet; ChannelIngress will find it
                        )
                    except Exception as e:
                        container.logger and container.logger.warning(
                            f"Slack download failed: {e}", exc_info=True
                        )

                file_refs.append(
                    {
                        "id": file_id,
                        "name": name,
                        "mimetype": mimetype,
                        "size": size,
                        "uri": uri,
                        "url_private": url_priv,
                        "platform": "slack",
                        "channel_key": ch_key,
                        "ts": ev.get("ts"),
                    }
                )

        # Turn Slack file_refs into IncomingFile so Ingress can do inbox + payload
        incoming_files: list[IncomingFile] = []
        for fr in file_refs:
            incoming_files.append(
                IncomingFile(
                    id=fr["id"],
                    name=fr["name"],
                    mimetype=fr.get("mimetype"),
                    size=fr.get("size"),
                    uri=fr.get("uri"),  # already artifact-backed
                    url=None,  # no re-download
                    extra={
                        "platform": "slack",
                        "channel_key": fr.get("channel_key"),
                        "ts": fr.get("ts"),
                    },
                )
            )

        meta = {
            "raw": payload,
            "channel_key": ch_key,
        }

        # Let ChannelIngress find the continuation, update inbox, and resume
        resumed = await ingress.handle(
            IncomingMessage(
                scheme=scheme,
                channel_id=channel_id,
                thread_id=str(thread_ts or ""),
                text=text,
                files=incoming_files or None,
                meta=meta,
            )
        )

        if container.logger:
            container.logger.for_run().debug(
                f"[Slack] inbound message: text={text!r}, files={len(incoming_files)}, resumed={resumed}"
            )

        # Nothing special to return to Slack (Events API only cares that we 200)
        return {}

    # --- file_shared (out-of-band file) ---
    if ev_type == "file_shared":
        team = payload.get("team_id")
        file_id = (ev.get("file") or {}).get("id")
        thread_ts = (
            (ev.get("file") or {}).get("thread_ts")
            or (ev.get("channel") or {}).get("thread_ts")
            or (ev.get("event_ts"))
        )
        chan = ev.get("channel_id") or (ev.get("channel") or {}).get("id")
        if not (file_id and chan):
            return {}

        ch_key = _channel_key(team, chan, None)
        scheme, channel_id = _slack_scheme_and_channel_id(team, chan)

        info = await c.slack.client.files_info(file=file_id)
        f = info.get("file") or {}
        name = f.get("name") or f.get("title") or "file"
        mimetype = f.get("mimetype")
        size = f.get("size")
        url_priv = f.get("url_private") or f.get("url_private_download")

        uri = None
        if url_priv and SLACK_BOT_TOKEN:
            try:
                data_bytes = await _download_slack_file(url_priv, SLACK_BOT_TOKEN)
                uri = await _stage_and_save(
                    c,
                    data=data_bytes,
                    file_id=file_id,
                    name=name,
                    ch_key=ch_key,
                    cont=None,
                )
            except Exception as e:
                container.logger and container.logger.for_run().warning(
                    f"Slack download failed: {e}", exc_info=True
                )

        # Build IncomingFile with pre-saved uri
        incoming_file = IncomingFile(
            id=file_id,
            name=name,
            mimetype=mimetype,
            size=size,
            uri=uri,  # already artifact-backed, no re-download when uri used in ingress
            url=None,
            extra={
                "platform": "slack",
                "channel_key": ch_key,
                "ts": ev.get("event_ts"),
            },
        )

        meta = {"raw": payload, "channel_key": ch_key}

        resumed = await ingress.handle(
            IncomingMessage(
                scheme=scheme,
                channel_id=channel_id,
                thread_id=str(thread_ts or ""),
                text="",  # no text; just a file drop
                files=[incoming_file],
                meta=meta,
            )
        )

        if container.logger:
            container.logger.for_run().debug(
                f"[Slack] file_shared: file_id={file_id}, resumed={resumed}"
            )

        return {}

    # other events might be added later
    return {}


async def handle_slack_interactive_common(container, payload: dict) -> dict:
    """
    Common handler for Slack interactive payloads (buttons, etc.).
    Can be called from HTTP /slack/interact or from Socket Mode.
    """
    c = container

    action = (payload.get("actions") or [{}])[0]
    team = (payload.get("team") or {}).get("id")
    chan = (payload.get("channel") or {}).get("id") or (payload.get("container") or {}).get(
        "channel_id"
    )
    # thread_ts = (payload.get("message") or {}).get("thread_ts")
    ch_key = _channel_key(team, chan, None)

    meta_raw = action.get("value") or "{}"
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {"choice": meta_raw}  # super defensive fallback

    choice = meta.get("choice", "reject")

    # value contains {"choice", "run_id", "node_id", "token", "sig" (optional)}
    token = meta.get("token")
    run_id = meta.get("run_id")
    node_id = meta.get("node_id")
    if token and run_id and node_id:
        await c.resume_router.resume(
            run_id=run_id,
            node_id=node_id,
            token=token,
            payload={
                "choice": choice,
                "slack_ts": (payload.get("message") or {}).get("ts"),
                "channel_key": ch_key,
            },
        )

    return {}
