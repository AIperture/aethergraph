from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import hmac
from typing import Any
from uuid import uuid4

import aiohttp

from aethergraph.services.channel.choices import normalize_choice_reply
from aethergraph.services.channel.resources import (
    ArtifactIngressScope,
    InputResource,
    InputResourceNormalizer,
    ResourceSet,
    ResourceStager,
)
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
    artifact_id: str | None = None  # AetherGraph artifact id when already materialized
    extra: dict[str, Any] = None  # Any extra metadata

    def __getitem__(self, item):
        return getattr(self, item, None)


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
    attachments: Iterable[InputResource | dict[str, Any]] | None = None

    # For approval
    choice: str | None = None  # User's choice/response
    conversation_id: str | None = None

    # Optional structured metadata
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class IngressPersistenceScope:
    """Event-log scope used for one canonical inbound message."""

    scope_id: str
    kind: str
    user_id: str | None = None
    org_id: str | None = None


@dataclass(frozen=True)
class ExpectedContinuation:
    """Exact continuation identity required by an interaction response."""

    token: str
    kind: str | None = None


@dataclass(frozen=True)
class InboundResult:
    """Result of accepting and optionally resuming one inbound message."""

    event_id: str | None
    resumed: bool


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
        self.normalizer = InputResourceNormalizer()

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

    def _conversation_id(
        self,
        *,
        msg: IncomingMessage,
        ch_key: str,
    ) -> str:
        if msg.conversation_id:
            return msg.conversation_id
        if msg.thread_id:
            return f"{ch_key}#thread:{msg.thread_id}"
        return ch_key

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
        mime: str | None,
        ch_key: str,
        conversation_id: str,
        cont: Continuation | None,
        source: str,
        meta: dict[str, Any] | None = None,
    ) -> InputResource:
        session_id = getattr(cont, "session_id", None) if cont else None
        run_id = None if session_id else (cont.run_id if cont else None)
        node_id = cont.node_id if cont else "resource_ingress"
        return await ResourceStager(container=self.c).stage_bytes(
            data,
            name=name,
            mime=mime,
            file_id=file_id,
            scope=ArtifactIngressScope(
                source=source,
                session_id=session_id,
                run_id=run_id,
                channel_key=ch_key,
                conversation_id=conversation_id,
                node_id=node_id,
                tool_name=f"{source}.resource_ingress",
            ),
            labels={"channel_key": ch_key},
            meta=meta,
        )

    async def _handle_files(
        self,
        msg: IncomingMessage,
        *,
        ch_key: str,
        conversation_id: str,
        cont: Continuation | None,
    ) -> tuple[ResourceSet, list[dict[str, Any]]]:
        """
        Normalize and optionally persist incoming files to artifact store.

        Returns a list of file_refs that mirror the Slack file_refs shape:
          {id, name, mimetype, size, uri, url, platform, channel_key, ...}
        """
        resources = ResourceSet()

        for attachment in msg.attachments or []:
            if isinstance(attachment, InputResource):
                resources.add(attachment)
            elif isinstance(attachment, dict):
                resources.add(self.normalizer.from_dict(attachment, source=msg.scheme))

        for f in msg.files or []:
            name = f.name or f.id or "unnamed"
            file_id = f.id or name
            mimetype = f.mimetype or "application/octet-stream"
            uri = f.uri
            url = f.url

            resource = self.normalizer.from_incoming_file(f, source=msg.scheme)
            if (not resource.artifact_id) and (not uri) and url:
                try:
                    data_bytes = await self._download_url(url)
                    resource = await self._stage_file(
                        data=data_bytes,
                        file_id=file_id,
                        name=name,
                        mime=mimetype,
                        ch_key=ch_key,
                        conversation_id=conversation_id,
                        cont=cont,
                        source=msg.scheme,
                        meta=f.extra or {},
                    )
                except Exception as e:
                    self._log("warning", f"Ingress: file download failed: {e}", channel_key=ch_key)

            resource.meta.setdefault("platform", msg.scheme)
            resource.meta.setdefault("channel_key", ch_key)
            resources.add(resource)

        resources.dedupe()
        file_refs = resources.to_display_files()

        # Append to per-channel inbox, dedup by id
        if file_refs and self.kv_hot is not None:
            inbox_key = f"inbox://{ch_key}"
            await self.kv_hot.list_append_unique(
                inbox_key,
                file_refs,
                id_key="id",
            )
        return resources, file_refs

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

    def _has_live_resume_target(self, cont: Continuation) -> bool:
        waits = getattr(self.c, "wait_registry", None)
        if waits is not None and hasattr(waits, "has") and waits.has(cont.token):
            return True

        sched_registry = getattr(self.c, "sched_registry", None)
        if sched_registry is not None and getattr(sched_registry, "get", None):
            return bool(sched_registry.get(cont.run_id))

        return False

    @staticmethod
    def _interaction_kind(cont: Continuation | None) -> str | None:
        if cont is None:
            return None
        if cont.kind in ("approval", "choice"):
            return "choice"
        return "text"

    @staticmethod
    def _expected_matches(
        cont: Continuation | None,
        expected: ExpectedContinuation | None,
    ) -> bool:
        if expected is None:
            return True
        if cont is None or not expected.token:
            return False
        if not hmac.compare_digest(cont.token, expected.token):
            return False
        if expected.kind is None:
            return True
        return ChannelIngress._interaction_kind(cont) == expected.kind

    async def _resolve_continuation(
        self,
        *,
        msg: IncomingMessage,
        ch_key: str,
    ) -> Continuation | None:
        cont = await self._find_continuation(
            scheme=msg.scheme,
            ch_key=ch_key,
            thread_id=msg.thread_id,
        )

        drop_stale = bool((msg.meta or {}).get("_drop_stale_continuation"))
        if drop_stale and cont and not self._has_live_resume_target(cont):
            self._log(
                "info",
                "Ingress: dropping stale continuation without live waiter/scheduler",
                channel_key=ch_key,
                run_id=cont.run_id,
                node_id=cont.node_id,
            )
            try:
                await self.cont_store.delete(cont.run_id, cont.node_id)
            except Exception as e:
                self._log(
                    "warning",
                    f"Ingress: failed to delete stale continuation: {e}",
                    channel_key=ch_key,
                    run_id=cont.run_id,
                    node_id=cont.node_id,
                )
            return None

        return cont

    async def _process(
        self,
        msg: IncomingMessage,
        *,
        persistence_scope: IngressPersistenceScope | None,
        expected_continuation: ExpectedContinuation | None,
        source_meta: dict[str, Any] | None,
    ) -> InboundResult:
        scheme = msg.scheme
        ch_key = self._channel_key(scheme, msg.channel_id)
        conversation_id = self._conversation_id(msg=msg, ch_key=ch_key)
        cont = await self._resolve_continuation(msg=msg, ch_key=ch_key)

        if not self._expected_matches(cont, expected_continuation):
            self._log(
                "info",
                "Ingress: expected continuation did not match",
                channel_key=ch_key,
            )
            return InboundResult(event_id=None, resumed=False)

        resources, file_refs = await self._handle_files(
            msg,
            ch_key=ch_key,
            conversation_id=conversation_id,
            cont=cont,
        )
        normalized_attachments = resources.to_attachment_dicts()
        normalized_choice: dict[str, Any] = {
            "choice": msg.choice,
            "choice_label": (msg.meta or {}).get("choice_label"),
            "text": msg.text or "",
            "matched": False,
        }
        if cont and cont.kind in ("approval", "choice"):
            normalized_choice = normalize_choice_reply(
                prompt=getattr(cont, "prompt", None),
                raw_choice=msg.choice,
                raw_text=msg.text or "",
            )

        payload: dict[str, Any] | None = None
        if cont:
            meta = {
                **(msg.meta or {}),
                "attachments": normalized_attachments,
            }
            if cont.kind in ("approval", "choice"):
                payload = {
                    **normalized_choice,
                    "channel_key": ch_key,
                    "conversation_id": conversation_id,
                    "thread_id": msg.thread_id,
                    "meta": meta,
                }
            elif cont.kind in ("user_files", "user_input_or_files"):
                payload = {
                    "text": msg.text or "",
                    "files": file_refs,
                    "attachments": normalized_attachments,
                    "channel_key": ch_key,
                    "conversation_id": conversation_id,
                    "thread_id": msg.thread_id,
                    "meta": meta,
                }
            else:
                payload = {
                    "text": msg.text or "",
                    "attachments": normalized_attachments,
                    "channel_key": ch_key,
                    "conversation_id": conversation_id,
                    "thread_id": msg.thread_id,
                    "meta": meta,
                }

        event_id: str | None = None
        if persistence_scope is not None and (
            msg.text or msg.choice or file_refs or normalized_attachments
        ):
            event_id = str(uuid4())
            choice = normalized_choice.get("choice")
            choice_label = normalized_choice.get("choice_label")
            display_text = msg.text or choice_label or choice or ""
            continuation_token = cont.token if cont else None
            interaction_kind = self._interaction_kind(cont)
            event_meta = {
                **(msg.meta or {}),
                **(source_meta or {}),
                "direction": "inbound",
                "role": "user",
                "source": scheme,
                "channel_key": ch_key,
                "conversation_id": conversation_id,
                "attachments": normalized_attachments,
                "choice": choice,
                "choice_label": choice_label,
                "continuation_token": continuation_token,
                "interaction_kind": interaction_kind,
            }
            row: dict[str, Any] = {
                "id": event_id,
                "ts": datetime.now(UTC).timestamp(),
                "scope_id": persistence_scope.scope_id,
                "kind": persistence_scope.kind,
                "payload": {
                    "type": "user.message",
                    "text": display_text,
                    "buttons": [],
                    "file": None,
                    "files": file_refs,
                    "attachments": normalized_attachments,
                    "choice": choice,
                    "choice_label": choice_label,
                    "continuation_token": continuation_token,
                    "interaction_kind": interaction_kind,
                    "meta": event_meta,
                },
            }
            if persistence_scope.user_id is not None:
                row["user_id"] = persistence_scope.user_id
            if persistence_scope.org_id is not None:
                row["org_id"] = persistence_scope.org_id
            await self.c.eventlog.append(row)

        if cont and payload is not None:
            await self.resume_router.resume(
                run_id=cont.run_id,
                node_id=cont.node_id,
                token=cont.token,
                payload=payload,
            )
            return InboundResult(event_id=event_id, resumed=True)

        self._log(
            "info",
            "Ingress: no continuation found for inbound message",
            channel_key=ch_key,
        )
        return InboundResult(event_id=event_id, resumed=False)

    # ---- Public method ----
    async def handle(self, msg: IncomingMessage) -> bool:
        """
        Handle an inbound message and resume a waiting continuation if any.

        Returns:
          True  -> a continuation was found and resumed
          False -> nothing was listening on this channel (fire-and-forget)
        """
        result = await self._process(
            msg,
            persistence_scope=None,
            expected_continuation=None,
            source_meta=None,
        )
        return result.resumed

    async def accept(
        self,
        msg: IncomingMessage,
        *,
        persistence_scope: IngressPersistenceScope,
        expected_continuation: ExpectedContinuation | None = None,
        source_meta: dict[str, Any] | None = None,
    ) -> InboundResult:
        """Persist one canonical inbound row and resume its exact continuation."""

        return await self._process(
            msg,
            persistence_scope=persistence_scope,
            expected_continuation=expected_continuation,
            source_meta=source_meta,
        )
