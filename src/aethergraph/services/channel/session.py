from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import logging
from typing import Any, Literal

from aethergraph.contracts.services.channel import Button, FileRef, OutEvent
from aethergraph.services.continuations.continuation import Correlator


class ChannelSession:
    """Helper to manage a channel-based session within a NodeContext.
    Provides methods to send messages, ask for user input or approval, and stream messages.
    The channel key is read from `session.channel` in the context.
    """

    def __init__(self, context, channel_key: str | None = None):
        self.ctx = context
        self._override_key = channel_key  # optional strong binding

    @property
    def _memory_facade(self):
        """
        Best-effort resolver for MemoryFacade.

        We intentionally go via ctx.services.memory_facade instead of ctx.memory()
        so that:
        - we reuse the same scoped facade NodeContext exposes, and
        - we do NOT raise if memory is not bound (auto logging should be optional).
        """
        return getattr(self.ctx.services, "memory_facade", None)

    # Channel bus
    @property
    def _bus(self):
        return self.ctx.services.channels

    # Continuation store
    @property
    def _cont_store(self):
        return self.ctx.services.continuation_store

    @property
    def _run_id(self):
        return self.ctx.run_id

    @property
    def _node_id(self):
        return self.ctx.node_id

    @property
    def _session_id(self):
        return self.ctx.session_id

    def _inject_context_meta(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Merge caller-provided meta with context-derived metadata
        (run_id, session_id, agent_id, app_id, graph_id, node_id).

        Caller-supplied keys win; we only fill in defaults.
        """
        base: dict[str, Any] = dict(meta or {})
        ctx = self.ctx

        # Use setdefault so explicit meta wins.
        if getattr(ctx, "run_id", None) is not None:
            base.setdefault("run_id", ctx.run_id)

        if getattr(ctx, "graph_id", None) is not None:
            base.setdefault("graph_id", ctx.graph_id)

        if getattr(ctx, "node_id", None) is not None:
            base.setdefault("node_id", ctx.node_id)

        if getattr(ctx, "session_id", None) is not None:
            base.setdefault("session_id", ctx.session_id)

        if getattr(ctx, "agent_id", None) is not None:
            base.setdefault("agent_id", ctx.agent_id)

        if getattr(ctx, "app_id", None) is not None:
            base.setdefault("app_id", ctx.app_id)

        return base

    def _default_chat_tags(
        self,
        extra: list[str] | None = None,
        *,
        channel: str | None = None,
    ) -> list[str]:
        """
        Derive some lightweight, structured tags from context
        and merge with caller-provided tags.
        """
        tags: list[str] = []

        # Channel name is very useful when debugging
        try:
            ch = self._resolve_key(channel)
            tags.append(f"channel:{ch}")
        except Exception:
            pass

        if self._run_id:
            tags.append(f"run:{self._run_id}")
        if self._session_id:
            tags.append(f"session:{self._session_id}")
        if self._node_id:
            tags.append(f"node:{self._node_id}")

        if extra:
            tags.extend(extra)

        return tags

    async def _log_chat(
        self,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
        enabled: bool = True,
        channel: str | None = None,
    ) -> None:
        """
        Internal helper: best-effort chat logging.
        Respects `enabled` and silently no-ops if memory is missing.
        """
        if not enabled or not text:
            return

        mem = self._memory_facade
        if not mem:
            return

        print(
            f"📝 ChannelSession: logging chat message to memory (role={role}) with message (text={text})"
        )
        await mem.record_chat(
            role,
            text,
            tags=self._default_chat_tags(tags, channel=channel),
            data=data,
            severity=severity,
            signal=signal,
        )

    def _resolve_default_key(self) -> str:
        """Unified default resolver (bus default → console)."""
        return self._bus.get_default_channel_key() or "console:stdin"

    def _resolve_key(self, channel: str | None = None) -> str:
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

    def _ensure_channel(self, event: "OutEvent", channel: str | None = None) -> "OutEvent":
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
    async def send(self, event: OutEvent, *, channel: str | None = None):
        """
        Send a single outbound event to the configured channel.

        This method ensures the event is associated with the correct channel,
        merges context-derived metadata, and publishes the event via the channel bus.
        This is the core low-level send method; higher-level convenience methods
        (e.g., `send_text`, `send_rich`, etc.) build on top of this and are recommended
        for common use cases.

        Examples:
            Basic usage to send a pre-constructed event:
            ```python

            event = OutEvent(type="agent.message", text="Hello!", channel=None)
            await context.channel().send(event)
            ```

            Sending to a specific channel:
            ```python
            await context.channel().send(event, channel="web:chat")
            ```

        Args:
            event: The `OutEvent` instance to send. If `event.channel` is not set,
                it will be resolved automatically.
            channel: Optional explicit channel key to override the default or event's channel.

        Returns:
            None

        Notes:
        for AG WebUI, you can set meta with
        ```python
            {
                "agent_id": "agent-123",
                "name": "Analyst",
            }
        ```
        to override the sender's display name and avatar in the chat.
        """
        event = self._ensure_channel(event, channel=channel)

        # merge context meta
        event.meta = self._inject_context_meta(event.meta)
        await self._bus.publish(event)

    async def send_text(
        self,
        text: str,
        *,
        meta: dict[str, Any] | None = None,
        channel: str | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,  # extra structured data
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send a plain text message to the configured channel.

        This method constructs a normalized outbound event, merges context-derived metadata,
        and dispatches the message via the channel bus.

        Examples:
            Basic usage to send a text message:
            ```python
            await context.channel().send_text("Hello, world!")
            ```

            Sending with additional metadata and to a specific channel:
            ```python
            await context.channel().send_text(
                "Status update.",
                meta={"priority": "high"},
                channel="web:chat"
            )
            ```

        Args:
            text: The primary text content to send.
            meta: Optional dictionary of metadata to include with the event.
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_log: Whether to log this message to memory (default: True).
            memory_role: The role to use when logging to memory (default: "assistant").
            memory_tags: Optional list of tags to associate with the memory log entry.
            memory_data: Optional structured data to include with the memory log entry.
            memory_severity: Severity level for the memory log entry (default: 2).
            memory_signal: Optional signal value for the memory log entry.

        Returns:
            None

        Notes:
        for AG WebUI, you can set meta with
        ```python
            {
                "agent_id": "agent-123",
                "name": "Analyst",
            }
        ```
        """

        await self._log_chat(
            memory_role,
            text,
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        event = OutEvent(
            type="agent.message",
            channel=self._resolve_key(channel),
            text=text,
            meta=self._inject_context_meta(meta),
        )
        await self._bus.publish(event)

    async def send_rich(
        self,
        text: str | None = None,
        *,
        rich: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        channel: str | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,  # extra structured data
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send a rich message to the configured channel.

        This method constructs and dispatches an outbound event that can include both plain text and
        structured rich content (such as cards, tables, or custom payloads). Context-derived metadata
        is automatically merged, and the event is published via the channel bus.

        Examples:
            Basic usage to send a rich message:
            ```python
            await context.channel().send_rich(
                text="Here are your results:",
                rich={"table": {"rows": [["A", 1], ["B", 2]]}}
            )
            ```

            Sending with additional metadata and to a specific channel:
            ```python
            await context.channel().send_rich(
                text="Task completed.",
                rich={"status": "success"},
                meta={"priority": "high"},
                channel="web:chat"
            )
            ```

        Args:
            text: The primary text content to send (optional).
            rich: A dictionary containing structured rich content to include with the message.
            meta: Optional dictionary of metadata to include with the event.
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_*: Parameters controlling logging to memory (see `send_text` for details).

        Returns:
            None

        Notes:
        for AG WebUI, you can set meta with
        ```python
            {
                "agent_id": "agent-123",
                "name": "Analyst",
            }
        ```
        """

        await self._log_chat(
            memory_role,
            text or "",
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        await self._bus.publish(
            OutEvent(
                type="agent.message",
                channel=self._resolve_key(channel),
                text=text,
                rich=rich,
                meta=self._inject_context_meta(meta),
            )
        )

    async def send_image(
        self,
        url: str | None = None,
        *,
        alt: str = "image",
        title: str | None = None,
        channel: str | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,  # extra structured data
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send an image message to the configured channel.

        This method constructs and dispatches an outbound event containing image metadata,
        including the image URL, alternative text, and an optional title. Context-derived
        metadata is automatically merged, and the event is published via the channel bus.

        Examples:
            Basic usage to send an image:
            ```python
            await context.channel().send_image(
                url="https://example.com/image.png",
                alt="Sample image"
            )
            ```

            Sending with a custom title and to a specific channel:
            ```python
            await context.channel().send_image(
                url="https://example.com/photo.jpg",
                alt="User profile photo",
                title="Profile",
                channel="web:chat"
            )
            ```

        Args:
            url: The URL of the image to send. If None, an empty string is used.
            alt: Alternative text describing the image (for accessibility).
            title: Optional title to display with the image.
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_*: Parameters controlling logging to memory (see `send_text` for details).

        Returns:
            None

        Notes:
            The capability to render images depends on the client adapter.
        """

        memory_tags = [*(memory_tags or []), "image"]
        await self._log_chat(
            memory_role,
            f"Image: {url or ''} (alt: {alt})",
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        await self._bus.publish(
            OutEvent(
                type="agent.message",
                channel=self._resolve_key(channel),
                text=title or alt,
                image={"url": url or "", "alt": alt, "title": title or ""},
                meta=self._inject_context_meta(None),
            )
        )

    async def send_file(
        self,
        url: str | None = None,
        *,
        file_bytes: bytes | None = None,
        filename: str = "file.bin",
        title: str | None = None,
        channel: str | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,  # extra structured data
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send a file to the configured channel in a normalized format.

        This method constructs and dispatches an outbound event containing file metadata,
        including the file URL, raw bytes, filename, and an optional title. Context-derived
        metadata is automatically merged, and the event is published via the channel bus.

        Examples:
            Basic usage to send a file by URL:
            ```python
            await context.channel().send_file(
                url="https://example.com/report.pdf",
                filename="report.pdf",
                title="Monthly Report"
            )
            ```

            Sending a file from bytes:
            ```python
            await context.channel().send_file(
                file_bytes=b"binarydata...",
                filename="data.bin",
                title="Raw Data"
            )
            ```

        Args:
            url: The URL of the file to send. If None, only file_bytes will be used.
            file_bytes: Optional raw bytes of the file to send.
            filename: The display name of the file (defaults to "file.bin").
            title: Optional title to display with the file.
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_*: Parameters controlling logging to memory (see `send_text` for details).

        Returns:
            None

        Notes:
            The capability to handle file uploads depends on the client adapter.
            If both `url` and `file_bytes` are provided, both will be included in the event.
        """
        file = {"filename": filename}
        if url:
            file["url"] = url
        if file_bytes is not None:
            file["bytes"] = file_bytes

        memory_tags = [*(memory_tags or []), "file"]
        await self._log_chat(
            memory_role,
            f"File: {filename} (url: {url or 'N/A'})",
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        await self._bus.publish(
            OutEvent(
                type="file.upload",
                channel=self._resolve_key(channel),
                text=title,
                file=file,
                meta=self._inject_context_meta(None),
            )
        )

    async def send_buttons(
        self,
        text: str,
        buttons: list[Button],
        *,
        meta: dict[str, Any] | None = None,
        channel: str | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,  # extra structured data
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send a message with interactive buttons to the configured channel.

        This method constructs and dispatches an outbound event containing a text prompt and a list of interactive buttons. Context-derived metadata is automatically merged, and the event is published via the channel bus.

        Examples:
            Basic usage to send a button prompt:
            ```python
            from aethergraph import Button
            await context.channel().send_buttons(
                "Choose an option:",
                [Button(label="Yes", value="yes"), Button(label="No", value="no")]
            )
            ```

            Sending with additional metadata and to a specific channel:
            ```python
            await context.channel().send_buttons(
                "Select your role:",
                [Button(label="Admin", value="admin"), Button(label="User", value="user")],
                meta={"priority": "high"},
                channel="web:chat"
            )
            ```

        Args:
            text: The primary text content to display above the buttons.
            buttons: A list of `Button` objects representing the interactive options.
            meta: Optional dictionary of metadata to include with the event.
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_*: Parameters controlling logging to memory (see `send_text` for details).

        Returns:
            None
        """
        memory_tags = [*(memory_tags or []), "buttons"]
        await self._log_chat(
            memory_role,
            text,
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        await self._bus.publish(
            OutEvent(
                type="link.buttons",
                channel=self._resolve_key(channel),
                text=text,
                buttons=buttons,
                meta=self._inject_context_meta(meta),
            )
        )

    # Small core helper to avoid the wait-before-resume race and DRY the flow.
    async def _ask_core(
        self,
        *,
        kind: str,
        payload: dict,  # what stored in continuation.payload
        channel: str | None,
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
                logger = logging.getLogger("aethergraph.services.channel.session")
                logger.debug("Continuation token %s already resolved inline", cont.token)
            try:
                await self._cont_store.delete(self._run_id, self._node_id)
            except Exception:
                logger.debug("Failed to delete continuation for token %s", cont.token)
                logger.exception("Error occurred while deleting continuation")
            return inline

        # 5) Push-only: bind correlator(s) so webhooks can locate the continuation
        corr = (res or {}).get("correlator")
        if corr:
            await self._cont_store.bind_correlator(token=cont.token, corr=corr)
            await self._cont_store.bind_correlator(  # message-less key for thread roots
                token=cont.token,
                corr=Correlator(
                    scheme=corr.scheme, channel=corr.channel, thread=corr.thread, message=""
                ),
            )
        else:
            # Best-effort binding (peek thread/channel)
            peek = await self._bus.peek_correlator(ch_key)
            if peek:
                await self._cont_store.bind_correlator(
                    token=cont.token, corr=Correlator(peek.scheme, peek.channel, peek.thread, "")
                )
            else:
                await self._cont_store.bind_correlator(
                    token=cont.token, corr=Correlator(self._bus._prefix(ch_key), ch_key, "", "")
                )

        # 6) Await the already-prepared future (router will resolve it later)
        return await fut

    # ------------------ Public ask_* APIs (race-free, normalized) ------------------
    async def ask_text(
        self,
        prompt: str | None,
        *,
        timeout_s: int = 3600,
        silent: bool = False,  # kept for back-compat; same behavior as before
        channel: str | None = None,
        # memory config
        memory_log_prompt: bool = True,
        memory_log_reply: bool = True,
        memory_tags: list[str] | None = None,
    ) -> str:
        """
        Prompt the user for a text response in a normalized format.

        This method sends a prompt to the configured channel, waits for a user reply, and returns the text input.
        It automatically handles context metadata, timeout, and channel resolution.

        Examples:
            Basic usage to prompt for user input:
            ```python
            reply = await context.channel().ask_text("What is your name?")
            ```

            Prompting with a custom timeout and silent mode:
            ```python
            reply = await context.channel().ask_text(
                "Enter your feedback.",
                timeout_s=120,
                silent=True
            )
            ```

        Args:
            prompt: The text prompt to display to the user. If None, a generic prompt may be shown.
            timeout_s: Maximum time in seconds to wait for a response (default: 3600).
            silent: If True, suppresses prompt display in some adapters (back-compat; default: False).
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_log_prompt: Whether to log the prompt to memory (default: True).
            memory_log_reply: Whether to log the user's reply to memory (default: True).
            memory_tags: Optional list of tags to associate with the memory log entries.

        Returns:
            str: The user's text response, or an empty string if no input was received.
        """
        if prompt:
            await self._log_chat(
                "assistant",
                prompt,
                tags=[*(memory_tags or []), "ask_text", "prompt"],
                enabled=memory_log_prompt,
                channel=channel,
            )

        payload = await self._ask_core(
            kind="user_input",
            payload={"prompt": prompt, "_silent": silent},
            channel=channel,
            timeout_s=timeout_s,
        )

        text = str(payload.get("text", ""))

        if text:
            await self._log_chat(
                "user",
                text,
                tags=[*(memory_tags or []), "ask_text", "reply"],
                enabled=memory_log_reply,
                channel=channel,
            )
        return text

    async def wait_text(
        self,
        *,
        timeout_s: int = 3600,
        channel: str | None = None,
        memory_log_reply: bool = True,
        memory_tags: list[str] | None = None,
    ) -> str:
        """
        Wait for a single text response from the user in a normalized format.

        This method prompts the user for input (with no explicit prompt), waits for a reply,
        and returns the text. It automatically handles context metadata, timeout, and channel resolution.

        Examples:
            Basic usage to wait for user input:
            ```python
            reply = await context.channel().wait_text()
            ```

            Waiting with a custom timeout and specific channel:
            ```python
            reply = await context.channel().wait_text(
                timeout_s=120,
                channel="web:chat"
            )
            ```

        Args:
            timeout_s: Maximum time in seconds to wait for a response (default: 3600).
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_log_reply: Whether to log the user's reply to memory (default: True).
            memory_tags: Optional list of tags to associate with the memory log entry.

        Returns:
            str: The user's text response, or an empty string if no input was received.
        """
        # Alias for ask_text(prompt=None) but keeps existing signature
        return await self.ask_text(
            prompt=None,
            timeout_s=timeout_s,
            silent=True,
            channel=channel,
            memory_log_prompt=False,  # no prompt to log
            memory_log_reply=memory_log_reply,
            memory_tags=memory_tags,
        )

    async def ask_approval(
        self,
        prompt: str,
        options: Iterable[str] = ("Approve", "Reject"),
        *,
        timeout_s: int = 3600,
        channel: str | None = None,
        memory_log_prompt: bool = True,
        memory_log_reply: bool = True,
        memory_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Prompt the user for approval or rejection in a normalized format.

        This method sends an approval prompt with customizable options (buttons) to the configured channel,
        waits for the user's selection, and returns a normalized result indicating approval status and choice.
        Context metadata, timeout, and channel resolution are handled automatically.

        Examples:
            Basic usage to prompt for approval:
            ```python
            result = await context.channel().ask_approval("Do you approve this action?")
            # result: { "approved": True/False, "choice": "Approve"/"Reject" }
            ```

            Prompting with custom options and timeout:
            ```python
            result = await context.channel().ask_approval(
                "Proceed with deployment?",
                options=["Yes", "No", "Defer"],
                timeout_s=120
            )
            ```

        Args:
            prompt: The text prompt to display to the user.
            options: Iterable of button labels for user choices (defaults to "Approve" and "Reject").
            timeout_s: Maximum time in seconds to wait for a response (default: 3600).
            channel: Optional explicit channel key to override the default or session-bound channel.
            memory_log_prompt: Whether to log the prompt to memory (default: True).
            memory_log_reply: Whether to log the user's reply to memory (default: True).
            memory_tags: Optional list of tags to associate with the memory log entries.

        Returns:
            dict: A dictionary containing:
                - "approved": bool indicating if the __first__ option was selected (True if approved, False otherwise).
                - "choice": The label of the button selected by the user (str or None).

        Warning:
            The returned "choices" are determined by the external adapter and may vary. To be robust, make sure
            to use `choices.lower()` and strip whitespace when comparing.
        """
        if prompt:
            await self._log_chat(
                "assistant",
                prompt,
                tags=[*(memory_tags or []), "ask_approval", "prompt"],
                enabled=memory_log_prompt,
                channel=channel,
            )

        payload = await self._ask_core(
            kind="approval",
            payload={"prompt": {"title": prompt, "buttons": list(options)}},
            channel=channel,
            timeout_s=timeout_s,
        )
        choice = payload.get("choice")
        if choice is not None:
            await self._log_chat(
                "user",
                f"Selected: {str(choice)}",
                tags=[*(memory_tags or []), "ask_approval", "reply"],
                enabled=memory_log_reply,
                channel=channel,
            )

        # Normalize return
        # 1) If adapter explicitly sets approved, trust it
        buttons = list(options)  # just plain list, not Button objects
        # 2) Fallback: derive from choice + options
        if choice is None or not buttons:
            approved = False
        else:
            choice_norm = str(choice).strip().lower()
            first_norm = str(buttons[0]).strip().lower()
            approved = choice_norm == first_norm

        return {
            "approved": approved,
            "choice": choice,
        }

    async def ask_files(
        self,
        prompt: str,
        *,
        accept: list[str] | None = None,
        multiple: bool = True,
        timeout_s: int = 3600,
        channel: str | None = None,
        memory_log_prompt: bool = True,
        memory_log_reply: bool = True,
        memory_tags: list[str] | None = None,
    ) -> dict:
        """
        Prompt the user to upload one or more files, optionally with a text comment.

        This method sends a file upload request to the configured channel, allowing the user to select files
        and optionally enter accompanying text. The `accept` parameter provides hints to the client UI about
        which file types are preferred, but is not enforced server-side. The method waits for the user's response
        and returns a normalized result containing both text and file references.

        Examples:
            Basic usage to prompt for file upload:
            ```python
            result = await context.channel().ask_files(
                prompt="Please upload your report."
            )
            # result: { "text": "...", "files": [FileRef(...), ...] }
            ```

            Restricting to images and allowing multiple files:
            ```python
            result = await context.channel().ask_files(
                prompt="Upload images for review.",
                accept=["image/png", ".jpg"],
                multiple=True
            )
            ```

        Args:
            prompt: The text prompt to display to the user above the file picker.
            accept: Optional list of MIME types or file extensions to suggest allowed file types (e.g., "image/png", ".pdf", ".jpg").
            multiple: If True, allows the user to select multiple files (default: True).
            timeout_s: Maximum time in seconds to wait for a response (default: 3600).
            channel: Optional explicit channel key to override the default or session-bound channel.

        Returns:
            dict: A dictionary containing:
                - "text": str, the user's comment or description (may be empty).
                - "files": List[FileRef], references to the uploaded files (empty if none).

        Notes:
            On console adapters, file upload is not supported; only text will be returned.
            The `accept` parameter is a UI hint and does not guarantee file type enforcement.
        """
        if prompt:
            await self._log_chat(
                "assistant",
                prompt,
                tags=[*(memory_tags or []), "ask_files", "prompt"],
                enabled=memory_log_prompt,
                channel=channel,
            )

        payload = await self._ask_core(
            kind="user_files",
            payload={"prompt": prompt, "accept": accept or [], "multiple": bool(multiple)},
            channel=channel,
            timeout_s=timeout_s,
        )

        text = str(payload.get("text", ""))
        if text:
            await self._log_chat(
                "user",
                text,
                tags=[*(memory_tags or []), "ask_files", "reply"],
                enabled=memory_log_reply,
                channel=channel,
            )

        return {
            "text": text,
            "files": payload.get("files", []) if isinstance(payload.get("files", []), list) else [],
        }

    async def ask_text_or_files(
        self,
        *,
        prompt: str,
        timeout_s: int = 3600,
        channel: str | None = None,
        memory_log_prompt: bool = True,
        memory_log_reply: bool = True,
        memory_tags: list[str] | None = None,
    ) -> dict:
        """
        Prompt the user for either text input or file uploads in a normalized format.
        This method sends a prompt to the configured channel, allowing the user to respond with
        text, files, or both. It waits for the user's response and returns a normalized result
        containing the text and file references.

        Not used very often; prefer ask_text + get_latest_uploads or ask_files.
        """
        if prompt:
            await self._log_chat(
                "assistant",
                prompt,
                tags=[*(memory_tags or []), "ask_text_or_files", "prompt"],
                enabled=memory_log_prompt,
                channel=channel,
            )

        payload = await self._ask_core(
            kind="user_input_or_files",
            payload={"prompt": prompt},
            channel=channel,
            timeout_s=timeout_s,
        )

        text = str(payload.get("text", ""))
        if text:
            await self._log_chat(
                "user",
                text,
                tags=[*(memory_tags or []), "ask_text_or_files", "reply"],
                enabled=memory_log_reply,
                channel=channel,
            )

        return {
            "text": text,
            "files": payload.get("files", []) if isinstance(payload.get("files", []), list) else [],
        }

    # ---------- inbox helpers (platform-agnostic) ----------
    async def get_latest_uploads(self, *, clear: bool = True) -> list[FileRef]:
        """
        Retrieve the latest uploaded files from this channel's inbox in a normalized format.

        This method accesses the ephemeral KV store to fetch file uploads associated with the current channel.
        By default, it clears the inbox after retrieval to prevent duplicate processing. If the KV service
        is unavailable in the context, a RuntimeError is raised. This method allows the fetch the files user
        uploaded __not__ from an ask_files prompt, but from any prior upload event.

        Examples:
            Basic usage to fetch and clear uploaded files:
            ```python
            files = await context.channel().get_latest_uploads()
            ```

            Fetching files without clearing the inbox:
            ```python
            files = await context.channel().get_latest_uploads(clear=False)
            ```

        Args:
            clear: If True (default), removes files from the inbox after retrieval.
                If False, files are returned but remain in the inbox.

        Returns:
            List[FileRef]: A list of `FileRef` objects representing the uploaded files.
                Returns an empty list if no files are present.

        Raises:
            RuntimeError: If the ephemeral KV service is not available in the current context.
        """
        kv = getattr(self.ctx.services, "kv", None)
        if kv:
            if clear:
                files = await kv.list_pop_all(self._inbox_kv_key) or []
            else:
                files = await kv.get(self._inbox_kv_key, []) or []
            return files
        else:
            raise RuntimeError(
                "EphemeralKV service not available in this context. Inbox not supported."
            )

    # ---------- streaming ----------
    class _StreamSender:
        def __init__(self, outer: "ChannelSession", *, channel_key: str | None = None):
            self._outer = outer
            self._started = False
            # Resolve once (explicit -> bound -> default)
            self._channel_key = outer._resolve_key(channel_key)
            self._upsert_key = f"{outer._run_id}:{outer._node_id}:stream"

        def _inject_context_meta(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            return self._outer._inject_context_meta(meta)

        def _buf(self):
            return getattr(self, "__buf", None)

        def _ensure_buf(self):
            if not hasattr(self, "__buf"):
                self.__buf = []
            return self.__buf

        async def start(self):
            if not self._started:
                self._started = True
                await self._outer._bus.publish(
                    OutEvent(
                        type="agent.stream.start",
                        channel=self._channel_key,
                        upsert_key=self._upsert_key,
                        meta=self._inject_context_meta(None),
                    )
                )

        async def delta(self, text_piece: str):
            await self.start()
            buf = self._ensure_buf()
            buf.append(text_piece)
            # Upsert full text so adapters can rewrite one message
            await self._outer._bus.publish(
                OutEvent(
                    type="agent.message.update",
                    channel=self._channel_key,
                    text="".join(buf),
                    upsert_key=self._upsert_key,
                    meta=self._inject_context_meta(None),
                )
            )

        async def end(
            self,
            full_text: str | None = None,
            *,
            # NEW: memory logging
            memory_log: bool = True,
            memory_role: Literal["assistant", "system", "tool", "user"] = "assistant",
            memory_tags: list[str] | None = None,
            memory_data: dict[str, Any] | None = None,
            memory_severity: int = 2,
            memory_signal: float | None = None,
        ):
            text_to_log = full_text
            if text_to_log is None:
                buf = self._buf()
                text_to_log = "".join(buf) if buf else ""

            # 1) Final UI update
            if full_text is not None:
                await self._outer._bus.publish(
                    OutEvent(
                        type="agent.message.update",
                        channel=self._channel_key,
                        text=full_text,
                        upsert_key=self._upsert_key,
                        meta=self._inject_context_meta(None),
                    )
                )

            # 2) Memory logging of the completed message
            await self._outer._log_chat(
                memory_role,
                text_to_log,
                tags=memory_tags,
                data=memory_data,
                severity=memory_severity,
                signal=memory_signal,
                enabled=memory_log,
                channel=self._channel_key,
            )

            # 3) End-of-stream event
            await self._outer._bus.publish(
                OutEvent(
                    type="agent.stream.end",
                    channel=self._channel_key,
                    upsert_key=self._upsert_key,
                    meta=self._inject_context_meta(None),
                )
            )

    @asynccontextmanager
    async def stream(self, channel: str | None = None) -> AsyncIterator["_StreamSender"]:
        """
        Stream a sequence of text deltas to the configured channel in a normalized format.

        This method provides a context manager for streaming incremental message updates (such as LLM generation)
        to the target channel. It automatically handles context metadata, upsert keys, and dispatches start, update,
        and end events to the channel bus. The caller is responsible for sending deltas and ending the stream.

        Examples:
            Basic usage to stream LLM output:
            ```python
            async with context.channel().stream() as s:
                await s.delta("Hello, ")
                await s.delta("world!")
                await s.end()
            ```

            Streaming to a specific channel:
            ```python
            async with context.channel().stream(channel="web:chat") as s:
                await s.delta("Generating results...")
                await s.end(full_text="Results complete.", memory_tags=["llm"])
            ```

        Args:
            channel: Optional explicit channel key to target a specific channel for this stream.
                If None, uses the session-bound or default channel.

        Returns:
            AsyncIterator[_StreamSender]: An async context manager yielding a `_StreamSender` object
                for sending deltas and ending the stream.

        Notes:
            The caller must explicitly call `end()` to finalize the stream. No auto-end is performed.
            The adapter may have specific behaviors for rendering streamed content (update vs. append).
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
            total: int | None = None,
            key_suffix: str = "progress",
            channel_key: str | None = None,
        ):
            self._outer = outer
            self._title = title
            self._total = total
            self._current = 0
            self._started = False
            # Resolve once (explicit -> bound -> default)
            self._channel_key = outer._resolve_key(channel_key)
            self._upsert_key = f"{outer._run_id}:{outer._node_id}:{key_suffix}"

        def _inject_context_meta(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            return self._outer._inject_context_meta(meta)

        async def start(self, *, subtitle: str | None = None):
            if not self._started:
                self._started = True
                await self._outer._bus.publish(
                    OutEvent(
                        type="agent.progress.start",
                        channel=self._channel_key,
                        upsert_key=self._upsert_key,
                        rich={
                            "title": self._title,
                            "subtitle": subtitle or "",
                            "total": self._total,
                            "current": self._current,
                        },
                        meta=self._inject_context_meta(None),
                    )
                )

        async def update(
            self,
            *,
            current: int | None = None,
            inc: int | None = None,
            subtitle: str | None = None,
            percent: float | None = None,
            eta_seconds: float | None = None,
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
            await self._outer._bus.publish(
                OutEvent(
                    type="agent.progress.update",
                    channel=self._channel_key,
                    upsert_key=self._upsert_key,
                    rich=payload,
                    meta=self._inject_context_meta(None),
                )
            )

        async def end(self, *, subtitle: str | None = "Done.", success: bool = True):
            await self._outer._bus.publish(
                OutEvent(
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
                    meta=self._inject_context_meta(None),
                )
            )

    @asynccontextmanager
    async def progress(
        self,
        *,
        title: str = "Working...",
        total: int | None = None,
        key_suffix: str = "progress",
        channel: str | None = None,
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
