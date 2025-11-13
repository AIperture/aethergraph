from typing import Any, Dict, Iterable, AsyncIterator, List, Optional

from aethergraph.contracts.services.channel import OutEvent, Button, FileRef
from aethergraph.services.continuations.continuation import Correlator

from contextlib import asynccontextmanager  

class ChannelSession:
    """Helper to manage a channel-based session within a NodeContext.
    Provides methods to send messages, ask for user input or approval, and stream messages.
    The channel key is read from `session.channel` in the context.
    """
    def __init__(self, context, channel_key: str | None = None):
        self.ctx = context
        self._override_key = channel_key  # optional strong binding

    # Channel bus 
    @property
    def _bus(self):         
        return self.ctx.services.channels

    # Continuation store
    @property
    def _cont_store(self): return self.ctx.services.continuation_store

    @property
    def _run_id(self): return self.ctx.run_id

    @property
    def _node_id(self): return self.ctx.node_id

    def _resolve_default_key(self) -> str:
        """Unified default resolver (bus default → console)."""
        return self._bus.get_default_channel_key() or "console:stdin"

    def _resolve_key(self, channel: Optional[str] = None) -> str:
        """
        Priority: explicit arg → bound override → resolved default,
        then run through ChannelBus alias resolver for canonical form.
        """
        raw = channel or self._override_key or self._resolve_default_key()
        if not raw:
            # Should never happen given the fallback, but fail fast if misconfigured
            raise RuntimeError("ChannelSession: unable to resolve a channel key")
        # NEW: alias → canonical resolution
        return self._bus.resolve_channel_key(raw)

    def _ensure_channel(self, event: "OutEvent", channel: Optional[str] = None) -> "OutEvent":
        """
        Ensure event.channel is set to a concrete channel key before publishing.
        If caller set event.channel already, keep it; otherwise fill in via resolver.
        """
        if not getattr(event, "channel", None):
            event.channel = self._resolve_key(channel)
        return event

    @property
    def _inbox_kv_key(self) -> str:
        """Key for this channel's inbox in ephemeral KV store (legacy helper)."""
        return f"inbox://{self._resolve_key()}"
    
    @property
    def _inbox_key(self) -> str:
        return f"inbox:{self._resolve_key()}" 

    
    # -------- send --------
    async def send(self, event: OutEvent, *, channel: Optional[str] = None):
        event = self._ensure_channel(event, channel=channel)
        await self._bus.publish(event)

    async def send_text(self, text: str, *, meta: Dict[str, Any] | None = None, channel: Optional[str] = None):
        event = OutEvent(type="agent.message", channel=self._resolve_key(channel), text=text, meta=meta or {})
        await self._bus.publish(event)

    async def send_rich(self, text: Optional[str] = None, *, rich: Optional[Dict[str, Any]] = None, meta: Dict[str, Any] | None = None, channel: Optional[str] = None):
        await self._bus.publish(OutEvent(type="agent.message", channel=self._resolve_key(channel), text=text, rich=rich, meta=meta or {}))

    async def send_image(self, url: Optional[str] = None, *, alt: str = "image", title: Optional[str] = None, channel: Optional[str] = None):
        await self._bus.publish(OutEvent(type="agent.message", channel=self._resolve_key(channel),
                                         text=title or alt, image={"url": url or "", "alt": alt, "title": title or ""}))
        
    async def send_file(self, url: Optional[str] = None, *, file_bytes: bytes | None = None, filename: str = "file.bin", title: Optional[str] = None, channel: Optional[str] = None):
        file = {"filename": filename}
        if url: file["url"] = url
        if file_bytes is not None: file["bytes"] = file_bytes
        await self._bus.publish(OutEvent(type="file.upload", channel=self._resolve_key(channel),
                                         text=title, file=file))

    async def send_buttons(self, text: str, buttons: List[Button], *, meta: Dict[str, Any] | None = None, channel: Optional[str] = None):
        await self._bus.publish(OutEvent(type="link.buttons", channel=self._resolve_key(channel),
                                         text=text, buttons=buttons, meta=meta or {}))

    # Small core helper to avoid the wait-before-resume race and DRY the flow.
    async def _ask_core(
        self,
        *,
        kind: str,
        payload: dict,                # what stored in continuation.payload
        channel: Optional[str],
        timeout_s: int,
    ) -> dict:
        ch_key = self._resolve_key(channel)

        # 1) Create continuation (with audit/security) 
        cont = await self.ctx.create_continuation(
            channel=ch_key, kind=kind, payload=payload, deadline_s=timeout_s
        )

        # 2) PREPARE the wait future BEFORE notifying (prevents race)
        fut = self.ctx.prepare_wait_for_resume(cont.token)

        # 3) Notify (console/local-web may return {"payload": ...} inline)
        res = await self._bus.notify(cont)

        # 4) Inline short-circuit: skip waiting and cleanup
        inline = (res or {}).get("payload")
        if inline is not None:
            # Defensive resolve (ok if already resolved by design)
            try:
                self.ctx.services.waits.resolve(cont.token, inline)
            except Exception:
                pass
            try:
                await self._cont_store.delete(self._run_id, self._node_id)
            except Exception:
                pass
            return inline

        # 5) Push-only: bind correlator(s) so webhooks can locate the continuation
        corr = (res or {}).get("correlator")
        if corr:
            await self._cont_store.bind_correlator(token=cont.token, corr=corr)
            await self._cont_store.bind_correlator(  # message-less key for thread roots
                token=cont.token,
                corr=Correlator(scheme=corr.scheme, channel=corr.channel, thread=corr.thread, message="")
            )
        else:
            # Best-effort binding (peek thread/channel)
            peek = await self._bus.peek_correlator(ch_key)
            if peek:
                await self._cont_store.bind_correlator(
                    token=cont.token,
                    corr=Correlator(peek.scheme, peek.channel, peek.thread, "")
                )
            else:
                await self._cont_store.bind_correlator(
                    token=cont.token,
                    corr=Correlator(self._bus._prefix(ch_key), ch_key, "", "")
                )

        # 6) Await the already-prepared future (router will resolve it later)
        return await fut


        
    # ------------------ Public ask_* APIs (race-free, normalized) ------------------
    async def ask_text(
        self,
        prompt: Optional[str],
        *,
        timeout_s: int = 3600,
        silent: bool = False,             # kept for back-compat; same behavior as before
        channel: Optional[str] = None
    ) -> str:
        payload = await self._ask_core(
            kind="user_input",
            payload={"prompt": prompt, "_silent": silent},
            channel=channel,
            timeout_s=timeout_s,
        )
        return str(payload.get("text", ""))


    async def wait_text(
        self,
        *,
        timeout_s: int = 3600,
        channel: Optional[str] = None
    ) -> str:
        # Alias for ask_text(prompt=None) but keeps existing signature
        return await self.ask_text(prompt=None, timeout_s=timeout_s, silent=True, channel=channel)


    async def ask_approval(
        self,
        prompt: str,
        options: Iterable[str] = ("Approve", "Reject"),
        *,
        timeout_s: int = 3600,
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = await self._ask_core(
            kind="approval",
            payload={"prompt": {"title": prompt, "buttons": list(options)}},
            channel=channel,
            timeout_s=timeout_s,
        )
        choice = payload.get("choice")

        # Normalize return
        # 1) If adapter explicitly sets approved, trust it
        buttons = list(options) # just plan list, not Button objects
        # 2) Fallback: derive from choice + options
        if choice is None or not buttons:
            approved = False
        else:
            choice_norm = str(choice).strip().lower()
            first_norm = str(buttons[0]).strip().lower()
            approved = (choice_norm == first_norm)

        return {
            "approved": approved,
            "choice": choice,
        }

    async def ask_files(
        self,
        *,
        prompt: str,
        accept: List[str] | None = None,
        multiple: bool = True,
        timeout_s: int = 3600,
        channel: Optional[str] = None
    ) -> dict:
        """
        Ask for file upload (plus optional text). Returns:
        { "text": str, "files": List[FileRef] }
        Note: console has no uploads; you’ll get only text there.

        The `accept` list can contain MIME types (e.g., "image/png") or file extensions (e.g., ".png"). This 
        is a hint to the client UI about what file types to accept. Aethergraph does not enforce file type restrictions.
        """
        payload = await self._ask_core(
            kind="user_files",
            payload={"prompt": prompt, "accept": accept or [], "multiple": bool(multiple)},
            channel=channel,
            timeout_s=timeout_s,
        )
        return {
            "text": str(payload.get("text", "")),
            "files": payload.get("files", []) if isinstance(payload.get("files", []), list) else []
        }


    async def ask_text_or_files(
        self,
        *,
        prompt: str,
        timeout_s: int = 3600,
        channel: Optional[str] = None
    ) -> dict:
        """
        Ask for either text or files. Returns:
        { "text": str, "files": List[FileRef] }
        """
        payload = await self._ask_core(
            kind="user_input_or_files",
            payload={"prompt": prompt},
            channel=channel,
            timeout_s=timeout_s,
        )
        return {
            "text": str(payload.get("text", "")),
            "files": payload.get("files", []) if isinstance(payload.get("files", []), list) else []
        }


    # ---------- inbox helpers (platform-agnostic) ----------
    async def get_latest_uploads(self, *, clear: bool = True) -> List[FileRef]:
        """Get latest uploaded files in this channel's inbox, optionally clearing them."""
        kv = getattr(self.ctx.services, "kv", None)
        if kv: 
            if clear:
                files = await kv.list_pop_all(self._inbox_kv_key) or []
            else:
                files = await kv.get(self._inbox_kv_key, []) or []
            return files 
        else:
            raise RuntimeError("EphemeralKV service not available in this context. Inbox not supported.")
    

    # ---------- streaming ----------
    class _StreamSender:
        def __init__(self, outer: "ChannelSession", *, channel_key: Optional[str] = None):
            self._outer = outer
            self._started = False
            # Resolve once (explicit -> bound -> default)
            self._channel_key = outer._resolve_key(channel_key)
            self._upsert_key = f"{outer._run_id}:{outer._node_id}:stream"

        def _buf(self): return getattr(self, "__buf", None)
        def _ensure_buf(self):
            if not hasattr(self, "__buf"): self.__buf = []
            return self.__buf

        async def start(self):
            if not self._started:
                self._started = True
                await self._outer._bus.publish(OutEvent(
                    type="agent.stream.start",
                    channel=self._channel_key,
                    upsert_key=self._upsert_key
                ))

        async def delta(self, text_piece: str):
            await self.start()
            buf = self._ensure_buf()
            buf.append(text_piece)
            # Upsert full text so adapters can rewrite one message
            await self._outer._bus.publish(OutEvent(
                type="agent.message.update",
                channel=self._channel_key,
                text="".join(buf),
                upsert_key=self._upsert_key
            ))

        async def end(self, full_text: Optional[str] = None):
            if full_text is not None:
                await self._outer._bus.publish(OutEvent(
                    type="agent.message.update",
                    channel=self._channel_key,
                    text=full_text,
                    upsert_key=self._upsert_key
                ))
            await self._outer._bus.publish(OutEvent(
                type="agent.stream.end",
                channel=self._channel_key,
                upsert_key=self._upsert_key
            ))

    @asynccontextmanager
    async def stream(self, channel: Optional[str] = None) -> AsyncIterator["_StreamSender"]:
        """
        Back-compat: no arg uses session/default/console.
        New: pass a channel key to target a specific channel for this stream.
        """
        s = ChannelSession._StreamSender(self, channel_key=channel)
        try:
            yield s
        finally:
            # No auto-end; caller decides when to end()
            pass

    # ---------- progress ----------
    class _ProgressSender:
        def __init__(
            self,
            outer: "ChannelSession",
            *,
            title: str = "Working...",
            total: Optional[int] = None,
            key_suffix: str = "progress",
            channel_key: Optional[str] = None,
        ):
            self._outer = outer
            self._title = title
            self._total = total
            self._current = 0
            self._started = False
            # Resolve once (explicit -> bound -> default)
            self._channel_key = outer._resolve_key(channel_key)
            self._upsert_key = f"{outer._run_id}:{outer._node_id}:{key_suffix}"

        async def start(self, *, subtitle: Optional[str] = None):
            if not self._started:
                self._started = True
                await self._outer._bus.publish(OutEvent(
                    type="agent.progress.start",
                    channel=self._channel_key,
                    upsert_key=self._upsert_key,
                    rich={
                        "title": self._title,
                        "subtitle": subtitle or "",
                        "total": self._total,
                        "current": self._current,
                    },
                ))

        async def update(
            self,
            *,
            current: Optional[int] = None,
            inc: Optional[int] = None,
            subtitle: Optional[str] = None,
            percent: Optional[float] = None,
            eta_seconds: Optional[float] = None
        ):
            await self.start()
            if percent is not None and self._total:
                self._current = int(round(self._total * max(0.0, min(1.0, percent))))
            if inc is not None:
                self._current += int(inc)
            if current is not None:
                self._current = int(current)
            payload = {
                "title": self._title,
                "subtitle": subtitle or "",
                "total": self._total,
                "current": self._current,
            }
            if eta_seconds is not None:
                payload["eta_seconds"] = float(eta_seconds)
            await self._outer._bus.publish(OutEvent(
                type="agent.progress.update",
                channel=self._channel_key,
                upsert_key=self._upsert_key,
                rich=payload,
            ))

        async def end(self, *, subtitle: Optional[str] = "Done.", success: bool = True):
            await self._outer._bus.publish(OutEvent(
                type="agent.progress.end",
                channel=self._channel_key,
                upsert_key=self._upsert_key,
                rich={
                    "title": self._title,
                    "subtitle": subtitle or "",
                    "success": bool(success),
                    "total": self._total,
                    "current": self._total if self._total is not None else None,
                },
            ))

    @asynccontextmanager
    async def progress(
        self,
        *,
        title: str = "Working...",
        total: Optional[int] = None,
        key_suffix: str = "progress",
        channel: Optional[str] = None
    ) -> AsyncIterator["_ProgressSender"]:
        """
        Back-compat: no channel uses session/default/console.
        New: pass channel to target a specific channel for this progress bar.
        """
        p = ChannelSession._ProgressSender(
            self, title=title, total=total, key_suffix=key_suffix, channel_key=channel
        )
        try:
            await p.start()
            yield p
        finally:
            # no auto-end
            pass
