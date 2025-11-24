from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import aiohttp

from aethergraph.services.continuations.continuation import Continuation, Correlator


@dataclass
class IncomingFile:
    """
    Generic description of a file coming from an external UI.

    You can:
      - pre-upload somewhere and pass url/uri, or
      - provide a public url and let AG download + store as artifact.
    """

    id: str | None = None  # Optional identifier for the file
    name: str | None = None  # Optional name of the file
    mimetype: str | None = None  # Optional MIME type of the file
    size: int | None = None  # Optional size of the file in bytes
    url: str | None = None  # URL where the file is located
    uri: str | None = None  # URI where the file is located
    extra: dict[str, Any] = None  # Any extra metadata


@dataclass
class IncomingMessage:
    """
    Transport-agnostic inbound message shape.
    Used by HTTP/WS handlers and any custom code that wants to resume via channel.
    """

    scheme: str  # e.g. "ext", "mychat", "slack-http", etc.
    channel_id: str  # Channel identifier
    thread_id: str | None = None  # Optional thread/conversation identifier

    # For ask_text / ask_file continuations
    text: str | None = None  # Text content of the message
    files: Iterable[IncomingFile] | None = None  # Attached files

    # For approval
    choice: str | None = None  # User's choice/response

    # Optional structured metadata
    meta: dict[str, Any] | None = None


class ChannelIngress:
    """
    Canonical entry point for inbound messages from external channels.

    Typical flow:
      UI -> HTTP/WS -> ChannelIngress.handle(...) -> cont_store + resume_router
    """

    def __init__(self, *, container, logger=None):
        self.c = container
        # Validate and assign dependencies

        assert container is not None, "Either provide all dependencies or a container"
        self.artifacts = container.artifacts if hasattr(container, "artifacts") else None
        self.kv_hot = container.kv_hot if hasattr(container, "kv_hot") else None
        self.cont_store = container.cont_store if hasattr(container, "cont_store") else None
        self.resume_router = (
            container.resume_router if hasattr(container, "resume_router") else None
        )

        if logger is not None:
            self.logger = logger
        else:
            container_logger = getattr(container, "logger", None)
            self.logger = container_logger.for_channel() if container_logger else None

    def _channel_key(self, scheme: str, channel_id: str) -> str:
        """
        Build a canonical channel key string from scheme + channel_id.

        - For the generic "ext" channel, we use "ext:chan/<id>".
        - For Slack/Telegram/etc. we can just use "<scheme>:<channel_id>" so we can
          preserve their existing formats.
        """
        if scheme == "ext":
            return f"{scheme}:chan/{channel_id}"
        # Slack: channel_id = "team/T:chan/C" => "slack:team/T:chan/C"
        # Telegram: channel_id = "chat/<id>[:topic/<topic_id>]" => "tg:chat/..."
        return f"{scheme}:{channel_id}"

    def _log(self, level: str, msg: str, **kwargs):
        if not self.logger:
            print(f"[{level.upper()}] {msg} | {kwargs}")
            return
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(msg, extra=kwargs)

    async def _download_url(self, url: str) -> bytes:
        """
        Simple downloader for public URLs.
        """
        async with aiohttp.ClientSession() as sess, sess.get(url) as r:
            r.raise_for_status()
            return await r.read()

    async def _stage_file(
        self,
        *,
        data: bytes,
        file_id: str | None,
        name: str,
        ch_key: str,
        cont: Continuation,
    ) -> str:
        """
        Write bytes to tmp path, then save via ArtifactStore.save_file(...).
        Returns the Artifact.uri (string).
        """
        tmp = await self.artifacts.plan_staging_path(planned_ext=f"_{file_id or name}")

        with open(tmp, "wb") as f:
            f.write(data)

        run_id = cont.run_id if cont else "ad-hoc"
        node_id = cont.node_id if cont else "channel-ingress"

        art = await self.artifacts.save_file(
            path=tmp,
            kind="upload",
            run_id=run_id,
            graph_id="channel",
            node_id=node_id,
            tool_name="channel.upload",
            tool_version="0.0.1",
            suggested_uri=None,
            pin=False,
            labels={
                "source": "channel",
                "channel": ch_key,
                "name": name,
                "inbound_file_id": file_id or "",
            },
            metrics=None,
            preview_uri=None,
        )

        saved_uri = getattr(art, "uri", None)
        if not saved_uri:
            self._log(
                "error",
                "Failed to save uploaded file as artifact",
                channel=ch_key,
            )

        return saved_uri

    async def _handle_files(
        self,
        msg: IncomingMessage,
        *,
        ch_key: str,
        cont: Continuation,
    ) -> list[dict[str, Any]]:
        """
        Normalize and optionally persist incoming files to artifact store.

        Returns a list of file_refs that mirror the Slack file_refs shape:
          {id, name, mimetype, size, uri, url, platform, channel_key, ...}
        """
        if not msg.files:
            return []

        file_refs: list[dict[str, Any]] = []
        for f in msg.files:
            name = f.name or f.id or "unnamed"
            file_id = f.id or name
            mimetype = f.mimetype or "application/octet-stream"
            size = f.size or 0
            uri = f.uri
            url = f.url

            # Optional: auto-download if url is provided and no uri
            # this is not executed when we stage files with channel-specific upload handlers that already provide uri
            if (not uri) and url:
                try:
                    data_bytes = await self._download_url(url)
                    uri = await self._stage_file(
                        data=data_bytes,
                        file_id=file_id,
                        name=name,
                        ch_key=ch_key,
                        cont=cont,
                    )
                except Exception as e:
                    self._log("warning", f"Ingress: file download failed: {e}", channel_key=ch_key)

            ref = {
                "id": file_id,
                "name": name,
                "mimetype": mimetype,
                "size": size,
                "uri": uri,
                "url": url,
                "platform": msg.scheme,
                "channel_key": ch_key,
            }
            if f.extra:
                ref["extra"] = dict(f.extra)

            file_refs.append(ref)

        # Append to per-channel inbox, dedup by id
        inbox_key = f"inbox://{ch_key}"
        await self.kv_hot.list_append_unique(
            inbox_key,
            file_refs,
            id_key="id",
        )
        return file_refs

    async def _find_continuation(
        self, *, scheme: str, ch_key: str, thread_id: str | None
    ) -> Continuation | None:
        """
        Find pending continuation for this channel/thread.
        """
        cont = None
        if thread_id:
            corr = Correlator(scheme=scheme, channel=ch_key, thread=thread_id, message="")
            cont = await self.cont_store.find_by_correlator(corr=corr)

        if not cont:
            # Fallback: look for any continuation for this channel
            corr2 = Correlator(scheme=scheme, channel=ch_key, thread="", message="")
            cont = await self.cont_store.find_by_correlator(corr=corr2)

        return cont

    # ---- Public method ----
    async def handle(self, msg: IncomingMessage) -> bool:
        """
        Handle an inbound message and resume a waiting continuation if any.

        Returns:
          True  -> a continuation was found and resumed
          False -> nothing was listening on this channel (fire-and-forget)
        """
        scheme = msg.scheme
        ch_key = self._channel_key(scheme, msg.channel_id)

        cont = await self._find_continuation(
            scheme=scheme,
            ch_key=ch_key,
            thread_id=msg.thread_id,
        )

        # Normalize and persist any attached files
        file_refs = []
        if msg.files:
            file_refs = await self._handle_files(
                msg,
                ch_key=ch_key,
                cont=cont,
            )

        if not cont:
            # No continuation found, log and return
            self._log(
                "info",
                "Ingress: no continuation found for inbound message",
                channel_key=ch_key,
            )
            return False

        # Build payload for resumption
        kind = cont.kind
        meta = msg.meta or {}

        if kind == "approval":
            choice = (msg.choice or (msg.text or "")).strip() or "reject"
            payload: dict[str, Any] = {
                "choice": choice,
                "channel_key": ch_key,
                "thread_id": msg.thread_id,
                "meta": meta,
            }
        elif kind in ("user_files", "user_input_or_files"):
            payload = {
                "text": msg.text or "",
                "files": file_refs,
                "channel_key": ch_key,
                "thread_id": msg.thread_id,
                "meta": meta,
            }
        else:
            payload = {
                "text": msg.text or "",
                "channel_key": ch_key,
                "thread_id": msg.thread_id,
                "meta": meta,
            }

        await self.resume_router.resume(
            run_id=cont.run_id,
            node_id=cont.node_id,
            token=cont.token,
            payload=payload,
        )
        return True
