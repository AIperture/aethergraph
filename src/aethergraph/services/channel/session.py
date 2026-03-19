from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
import inspect
import logging
from pathlib import Path, PurePath
import time
from typing import Any, Literal
import uuid

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.services.channel import Button, FileRef, OutEvent
from aethergraph.services.continuations.continuation import Correlator
from aethergraph.services.tracing import resolve_tracer


def _artifact_filename(artifact: Artifact, fallback: str | None = None) -> str:
    labels = artifact.labels or {}
    if "filename" in labels and labels["filename"]:
        return str(labels["filename"])

    # If no explicit filename label, try URI
    if artifact.uri:
        try:
            return PurePath(artifact.uri).name or fallback or artifact.artifact_id
        except Exception:
            pass

    return fallback or artifact.artifact_id


def _artifact_to_chat_file(
    artifact: Artifact,
    fallback_filename: str | None = None,
) -> dict[str, Any]:
    labels = artifact.labels or {}

    filename = (
        labels.get("filename")
        or (PurePath(artifact.uri).name if artifact.uri else None)
        or fallback_filename
        or artifact.artifact_id
    )

    # 👇 Renderer hint from labels (e.g. {"renderer": "image"})
    renderer = labels.get("renderer")

    # Make sure we actually set a mimetype if possible
    mime = artifact.mime
    if not mime and filename:
        # crude but fine: infer from extension
        lower = filename.lower()
        if lower.endswith(".png"):
            mime = "image/png"
        elif lower.endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif lower.endswith(".gif"):
            mime = "image/gif"

    size = (
        getattr(artifact, "bytes", None)
        or getattr(artifact, "size", None)
        or getattr(artifact, "size_bytes", None)
    )

    return {
        "artifact_id": artifact.artifact_id,
        "name": filename,
        "filename": filename,
        "mimetype": mime,
        "size": size,
        "uri": artifact.artifact_id,
        "renderer": renderer,  # key for the UI
    }


def _image_filename(title: str | None, alt: str | None) -> str:
    base = title or alt or "image"
    p = Path(base)
    if p.suffix:
        return base
    return base + ".png"


class ChannelSession:
    """Helper to manage a channel-based session within a NodeContext.
    Provides methods to send messages, ask for user input or approval, and stream messages.
    The channel key is read from `session.channel` in the context.
    """

    def __init__(self, context, channel_key: str | None = None):
        self.ctx = context
        self._override_key = channel_key  # optional strong binding
        self._phase_group_id: str | None = None
        self._phase_seq: int = 0
        self._phase_group_counter: int = 0

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

    @property
    def _tracer(self):
        return resolve_tracer(getattr(self.ctx.services, "tracer", None))

    def _begin_reply_lifecycle(self) -> str:
        self._phase_group_counter += 1
        self._phase_group_id = f"{self._run_id}:{self._node_id}:phase-group:{self._phase_group_counter}:{uuid.uuid4().hex[:8]}"
        return self._phase_group_id

    def _ensure_reply_lifecycle(self) -> str:
        return self._phase_group_id or self._begin_reply_lifecycle()

    def _close_reply_lifecycle(self) -> None:
        self._phase_group_id = None
        # Note: _phase_seq is intentionally NOT reset here.
        # It is a monotonic counter for unique phase_event_ids across the run.

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
        Publish one outbound event after channel/meta normalization.

        Ensure `event.channel` is set when missing and merge context metadata
        into `event.meta` before publishing through the channel bus.

        Examples:
            Publish a pre-constructed event:
            ```python
            event = OutEvent(type="agent.message", text="Hello!", channel=None)
            await context.channel().send(event)
            ```

            Resolve the channel when `event.channel` is empty:
            ```python
            await context.channel().send(event, channel="web:chat")
            ```

        Args:
            event: Event to publish.
            channel: Optional fallback channel key used only when `event.channel` is unset.

        Returns:
            None: Complete when the event is published.

        Notes:
            Existing `event.channel` is preserved and not overridden by `channel`.
        """
        event = self._ensure_channel(event, channel=channel)

        # merge context meta
        event.meta = self._inject_context_meta(event.meta)
        await self._bus.publish(event)

    async def send_phase(
        self,
        phase: str,
        status: Literal["pending", "active", "done", "failed", "skipped"],
        *,
        label: str | None = None,
        detail: str | None = None,
        channel: str | None = None,
    ) -> None:
        """
        Emit a phase-style progress update event.

        Phases are decoupled from the reply lifecycle. Each call creates a
        unique timeline entry (no upsert) so the frontend can accumulate them
        as pending phases until a message claims them.

        Examples:
            Send a pending phase update:
            ```python
            await context.channel().send_phase("routing", "pending")
            ```

            Send an active phase update with details:
            ```python
            await context.channel().send_phase(
                "planning",
                "active",
                label="Planning Phase",
                detail="Calculating optimal routes",
            )
            ```

        Args:
            phase: Logical phase identifier.
            status: Phase status value. One of `"pending"`, `"active"`, `"done"`, `"failed"`, or `"skipped"`.
            label: Optional display label. Defaults to `phase.title()`.
            detail: Optional detail text. Defaults to an empty string.
            channel: Optional target channel key.

        Returns:
            None: Complete when the progress update is published.

        Notes:
            - Payload shape is UI-oriented and adapters may render or ignore it differently.
            - The frontend shows LivePulse while a phase is "active" and hides it on terminal status.
            - Phases accumulate until the next terminal message claims them for InlinePhaseBlock display.
        """
        ch_key = self._resolve_key(channel)
        self._phase_seq += 1
        phase_seq = self._phase_seq
        phase_updated_at = time.time()
        # Each emission is unique (no upsert — every call is a new timeline entry)
        phase_event_id = f"{self._run_id}:{self._node_id}:phase:{phase_seq}"

        rich = {
            "kind": "phase",
            "phase": phase,
            "status": status,
            "label": label or phase.title(),
            "detail": detail or "",
            "phase_seq": phase_seq,
            "phase_event_id": phase_event_id,
            "phase_updated_at": phase_updated_at,
        }

        await self._bus.publish(
            OutEvent(
                type="agent.progress.update",
                channel=ch_key,
                # No upsert_key — each emission creates a new event
                rich=rich,
                meta=self._inject_context_meta(
                    {
                        "kind": "phase",
                        "phase": phase,
                        "status": status,
                    }
                ),
            )
        )

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
        Send one plain-text message event and optionally log it to memory.

        Log the text via memory facade when enabled, then publish
        `OutEvent(type="agent.message")` with merged metadata.

        Examples:
            Send a message:
            ```python
            await context.channel().send_text("Hello, world!")
            ```

            Send to a specific channel with extra metadata:
            ```python
            await context.channel().send_text(
                "Status update.",
                meta={"priority": "high"},
                channel="web:chat"
            )
            ```

        Args:
            text: Message body to send.
            meta: Optional outbound event metadata.
            channel: Optional target channel key.
            memory_log: Enable chat-memory logging for this call.
            memory_role: Role used for the memory record.
            memory_tags: Optional tags for the memory record.
            memory_data: Optional structured data for the memory record.
            memory_severity: Severity value for memory logging.
            memory_signal: Optional signal value for memory logging.

        Returns:
            None: Complete when memory logging and publish steps finish.

        Notes:
            Set adapter-specific display hints in `meta` (for example `name` or `agent_id`).
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
            meta=self._inject_context_meta(
                {
                    **(meta or {}),
                    "phase_group_id": self._ensure_reply_lifecycle(),
                }
            ),
        )
        await self._bus.publish(event)
        self._close_reply_lifecycle()

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
        Send a message with an optional rich payload and optional memory logging.

        Publish `OutEvent(type="agent.message")` with both `text` and `rich`
        fields, and record chat memory when enabled.

        Examples:
            Send one rich block:
            ```python
            await context.channel().send_rich(
                text="Here is the loss curve:",
                rich={
                    "kind": "plot",
                    "title": "Training loss",
                    "payload": {"engine": "vega-lite", "spec": loss_vega_spec},
                },
            )
            ```

            Send multiple rich blocks:
            ```python
            await context.channel().send_rich(
                text="Training summary:",
                rich={"blocks": [{"kind": "plot"}, {"kind": "metrics"}]},
            )
            ```

        Args:
            text: Optional plain text content.
            rich: Optional structured payload for UI-aware adapters.
            meta: Optional outbound event metadata.
            channel: Optional target channel key.
            memory_log: Enable chat-memory logging for this call.
            memory_role: Role used for the memory record.
            memory_tags: Optional tags for the memory record.
            memory_data: Optional structured data for the memory record.
            memory_severity: Severity value for memory logging.
            memory_signal: Optional signal value for memory logging.

        Returns:
            None: Complete when memory logging and publish steps finish.

        Notes:
            If `text` is `None`, memory logging records `"[rich message]"`.
        """

        # --- 1) Memory logging (log *something* even if text=None) ---
        log_text = text or "[rich message]"
        await self._log_chat(
            memory_role,
            log_text,
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        # --- 2) Publish outbound event ---
        await self._bus.publish(
            OutEvent(
                type="agent.message",
                channel=self._resolve_key(channel),
                text=text,
                rich=rich,
                meta=self._inject_context_meta(
                    {
                        **(meta or {}),
                        "phase_group_id": self._ensure_reply_lifecycle(),
                    }
                ),
            )
        )
        self._close_reply_lifecycle()

    async def send_image(
        self,
        url: str | None = None,
        *,
        file_bytes: bytes | None = None,
        alt: str = "image",
        title: str | None = None,
        channel: str | None = None,
        artifact_labels: dict[str, Any] | None = None,
        # memory logging...
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send an image attachment via `send_file()` with image defaults.

        Build image-oriented labels and filename, then delegate to `send_file`
        using `artifact_kind="image"`.

        Examples:
            Send by URL:
            ```python
            await context.channel().send_image(
                url="https://example.com/image.png",
                alt="Sample image"
            )
            ```

            Send generated bytes:
            ```python
            await context.channel().send_image(
                file_bytes=b"...",
                alt="Generated image",
                title="Result"
            )
            ```

        Args:
            url: Optional HTTP(S) URL or local path for the image source.
            file_bytes: Optional image bytes. Preferred when both are provided.
            alt: Alternate text used for fallback titling/filename.
            title: Optional display title.
            channel: Optional target channel key.
            artifact_labels: Optional extra artifact labels.
            memory_log: Enable chat-memory logging for this call.
            memory_role: Role used for the memory record.
            memory_tags: Optional tags for the memory record.
            memory_data: Optional structured data for the memory record.
            memory_severity: Severity value for memory logging.
            memory_signal: Optional signal value for memory logging.

        Returns:
            None: Complete when delegated file send finishes.

        Notes:
            `artifact_labels` are merged with `{"renderer": "image"}`.
        """

        labels = {"renderer": "image"}
        if artifact_labels:
            labels.update(artifact_labels)

        # Reuse memory logging text
        memory_tags = [*(memory_tags or []), "image"]

        await self.send_file(
            url=url,
            file_bytes=file_bytes,
            filename=_image_filename(title, alt),
            title=title or alt,
            channel=channel,
            artifact_kind="image",
            artifact_labels=labels,
            memory_log=memory_log,
            memory_role=memory_role,
            memory_tags=memory_tags,
            memory_data=memory_data,
            memory_severity=memory_severity,
            memory_signal=memory_signal,
        )

    async def send_file(
        self,
        url: str | None = None,
        *,
        file_bytes: bytes | None = None,
        filename: str = "file.bin",
        title: str | None = None,
        channel: str | None = None,
        # NEW: optional hints for artifact
        artifact_kind: str = "file",
        artifact_labels: dict[str, Any] | None = None,
        # memory logging handled separately
        memory_log: bool = True,
        memory_role: Literal["user", "assistant", "system", "tool"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ):
        """
        Send a file attachment message, optionally persisting it as an artifact.

        Prefer artifact-backed payloads for `file_bytes` or local-path inputs.
        Fall back to a URL-only payload when artifact persistence is unavailable.

        Examples:
            Send by URL:
            ```python
            await context.channel().send_file(
                url="https://example.com/report.pdf",
                filename="report.pdf",
                title="Monthly Report"
            )
            ```

            Send from bytes:
            ```python
            await context.channel().send_file(
                file_bytes=b"binarydata...",
                filename="data.bin",
                title="Raw Data"
            )
            ```

        Args:
            url: Optional source URL or local filesystem path.
            file_bytes: Optional raw file bytes.
            filename: Display filename for the chat attachment.
            title: Optional message text. Defaults to `filename`.
            channel: Optional target channel key.
            artifact_kind: Artifact kind when writing to artifact store.
            artifact_labels: Optional labels attached to persisted artifacts.
            memory_log: Enable chat-memory logging for this call.
            memory_role: Role used for the memory record.
            memory_tags: Optional tags for the memory record.
            memory_data: Optional structured data for the memory record.
            memory_severity: Severity value for memory logging.
            memory_signal: Optional signal value for memory logging.

        Returns:
            None: Complete when the outbound message event is published.

        Notes:
            When both `file_bytes` and `url` are provided, bytes are attempted first.
        """
        # ------------------------------
        # 1) Maybe create an Artifact
        # ------------------------------
        chat_file: dict[str, Any] = {
            "name": filename,
        }

        artifact = None

        # Ensure labels always carry filename
        effective_labels: dict[str, Any] = dict(artifact_labels or {})
        effective_labels.setdefault("filename", filename)

        # Case A: raw bytes → stream to ArtifactStore
        if file_bytes is not None:
            try:
                artifacts = self.ctx.artifacts()
                async with artifacts.writer(
                    kind=artifact_kind,
                    planned_ext=Path(filename).suffix or None,
                    pin=False,
                ) as w:
                    write_result = w.write(file_bytes)
                    if inspect.isawaitable(write_result):
                        await write_result

                    add_labels = getattr(w, "add_labels", None)
                    if callable(add_labels):
                        add_labels(effective_labels)

                artifact = artifacts.last_artifact

            except Exception:
                import logging

                logging.getLogger("aethergraph.channel").exception("send_file_artifact_failed")

        # Case B: local path (non-HTTP) → save_file
        elif url and not url.startswith(("http://", "https://")):
            try:
                artifacts = self.ctx.artifacts()
                artifact = await artifacts.save_file(
                    path=url,
                    kind=artifact_kind,
                    labels=effective_labels,
                    name=filename,
                    pin=False,
                )
            except Exception:
                import logging

                logging.getLogger("aethergraph.channel").exception("send_file_save_failed")

        # ------------------------------
        # 1b) Normalize chat_file from artifact or fallback URL
        # ------------------------------
        if artifact is not None:
            # Use artifact meta → url, mimetype, renderer (from labels), etc.
            chat_file.update(_artifact_to_chat_file(artifact, fallback_filename=filename))
        else:
            # Fallback: just pass whatever URL we got (may be remote HTTP or None)
            if url:
                chat_file["url"] = url

            # If caller passed renderer in labels, keep honoring it
            if artifact_labels and "renderer" in artifact_labels:
                chat_file["renderer"] = artifact_labels["renderer"]

        # For compatibility with existing payloads that used "filename"
        chat_file.setdefault("filename", chat_file.get("name"))

        # ------------------------------
        # 2) Log to memory
        # ------------------------------
        memory_tags = [*(memory_tags or []), "file"]
        await self._log_chat(
            memory_role,
            f"File: {filename} (url: {chat_file.get('url') or 'N/A'})",
            tags=memory_tags,
            data=memory_data,
            severity=memory_severity,
            signal=memory_signal,
            enabled=memory_log,
            channel=channel,
        )

        # ------------------------------
        # 3) Publish OutEvent
        # ------------------------------
        await self._bus.publish(
            OutEvent(
                type="agent.message",  # UI treats as normal message with attachment
                channel=self._resolve_key(channel),
                text=title or filename,
                file=chat_file,
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
        Send a button prompt event and optionally log prompt text to memory.

        Publish `OutEvent(type="link.buttons")` with button definitions and
        merged metadata for the resolved channel.

        Examples:
            Send a yes/no prompt:
            ```python
            from aethergraph import Button
            await context.channel().send_buttons(
                "Choose an option:",
                [Button(label="Yes", value="yes"), Button(label="No", value="no")]
            )
            ```

            Send with custom metadata:
            ```python
            await context.channel().send_buttons(
                "Select your role:",
                [Button(label="Admin", value="admin"), Button(label="User", value="user")],
                meta={"priority": "high"},
                channel="web:chat"
            )
            ```

        Args:
            text: Prompt text displayed above the buttons.
            buttons: Button definitions to send.
            meta: Optional outbound event metadata.
            channel: Optional target channel key.
            memory_log: Enable chat-memory logging for this call.
            memory_role: Role used for the memory record.
            memory_tags: Optional tags for the memory record.
            memory_data: Optional structured data for the memory record.
            memory_severity: Severity value for memory logging.
            memory_signal: Optional signal value for memory logging.

        Returns:
            None: Complete when memory logging and publish steps finish.

        Notes:
            Button rendering/interaction semantics depend on the active adapter.
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
        span = await self._tracer.start_span(
            service="channel",
            operation=f"_ask_core:{kind}",
            request={
                "kind": kind,
                "payload": payload,
                "channel_key": ch_key,
                "timeout_s": timeout_s,
            },
            tags=["channel", "wait", kind],
            metadata=self._inject_context_meta({"channel_key": ch_key}),
        )
        try:
            resumed = self._take_matching_resume_payload(kind=kind, expected_payload=payload)
            if resumed is not None:
                await span.resume(
                    metadata=self._inject_context_meta({"channel_key": ch_key}),
                    response=resumed,
                )
                await span.finish(
                    response=resumed,
                    metadata=self._inject_context_meta({"channel_key": ch_key}),
                )
                return resumed

            cont_payload = {
                "_channel_wait_kind": kind,
                **payload,
            }
            cont = await self.ctx.create_continuation(
                channel=ch_key, kind=kind, payload=cont_payload, deadline_s=timeout_s
            )
            fut = self.ctx.prepare_wait_for_resume(cont.token)
            wait_meta = self._inject_context_meta(
                {"channel_key": ch_key, "continuation_token": cont.token}
            )
            await span.wait(metadata=wait_meta, request={"kind": kind, "payload": cont_payload})

            res = await self._bus.notify(cont)
            inline = (res or {}).get("payload")
            if inline is not None:
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
                await span.resume(metadata=wait_meta, response=inline)
                await span.finish(response=inline, metadata=wait_meta)
                return inline

            corr = (res or {}).get("correlator")
            if corr:
                await self._cont_store.bind_correlator(token=cont.token, corr=corr)
                await self._cont_store.bind_correlator(
                    token=cont.token,
                    corr=Correlator(
                        scheme=corr.scheme, channel=corr.channel, thread=corr.thread, message=""
                    ),
                )
            else:
                peek = await self._bus.peek_correlator(ch_key)
                if peek:
                    await self._cont_store.bind_correlator(
                        token=cont.token,
                        corr=Correlator(peek.scheme, peek.channel, peek.thread, ""),
                    )
                else:
                    await self._cont_store.bind_correlator(
                        token=cont.token, corr=Correlator(self._bus._prefix(ch_key), ch_key, "", "")
                    )

            result = await fut
            await span.resume(metadata=wait_meta, response=result)
            await span.finish(response=result, metadata=wait_meta)
            return result
        except Exception as exc:
            await span.fail(exc, metadata=self._inject_context_meta({"channel_key": ch_key}))
            raise

    def _take_matching_resume_payload(
        self,
        *,
        kind: str,
        expected_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Consume a replay resume payload when this node is being re-entered after a
        scheduler-backed resume.

        Cooperative waits normally suspend on an in-process Future, but after a replay
        the node body can be re-executed with `context.resume_payload` already attached.
        In that case we should return the payload directly instead of minting a fresh
        continuation and waiting again.
        """
        if getattr(self.ctx, "_channel_resume_payload_consumed", False):
            return None

        resume_payload = getattr(self.ctx, "resume_payload", None)
        if not isinstance(resume_payload, dict) or not resume_payload:
            return None

        resume_kind = resume_payload.get("_channel_wait_kind")
        if resume_kind is not None and resume_kind != kind:
            return None

        expected_prompt = expected_payload.get("prompt")
        if "prompt" in resume_payload and resume_payload.get("prompt") != expected_prompt:
            return None

        for key in ("accept", "multiple"):
            if key in expected_payload and resume_payload.get(key) != expected_payload.get(key):
                return None

        self.ctx._channel_resume_payload_consumed = True
        return resume_payload

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
        Prompt for user text and return normalized reply text.

        Optionally log prompt/reply to memory, then use the continuation flow to
        await `kind="user_input"` and return `payload["text"]` as `str`.

        Examples:
            Ask a question:
            ```python
            reply = await context.channel().ask_text("What is your name?")
            ```

            Use timeout and silent mode:
            ```python
            reply = await context.channel().ask_text(
                "Enter your feedback.",
                timeout_s=120,
                silent=True
            )
            ```

        Args:
            prompt: Prompt text. `None` means wait without showing a prompt.
            timeout_s: Continuation deadline in seconds.
            silent: Back-compat flag forwarded in continuation payload.
            channel: Optional target channel key.
            memory_log_prompt: Enable memory logging for the prompt.
            memory_log_reply: Enable memory logging for non-empty reply text.
            memory_tags: Optional tags applied to memory entries.

        Returns:
            str: User text reply, or `""` when absent.

        Notes:
            Reply memory logging occurs only when returned text is non-empty.
        """
        channel_key = self._resolve_key(channel)
        span = await self._tracer.start_span(
            service="channel",
            operation="ask_text",
            request={"prompt": prompt, "timeout_s": timeout_s, "channel_key": channel_key},
            tags=["channel", "ask", "user_input"],
            metadata=self._inject_context_meta({"channel_key": channel_key}),
        )
        try:
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
            await span.finish(
                response={"text": text},
                metadata=self._inject_context_meta({"channel_key": channel_key}),
            )
            return text
        except Exception as exc:
            await span.fail(exc, metadata=self._inject_context_meta({"channel_key": channel_key}))
            raise

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
        Prompt for a button-style approval choice and return normalized result.

        Send an `approval` continuation with button labels, wait for the user's
        selected choice, preserve any typed text, then derive `approved` from
        whether the first option was selected (case-insensitive).

        Examples:
            Use default approve/reject options:
            ```python
            result = await context.channel().ask_approval("Do you approve this action?")
            # result: { "approved": True/False, "choice": "Approve"/"Reject", "text": "" }
            ```

            Use custom options:
            ```python
            result = await context.channel().ask_approval(
                "Proceed with deployment?",
                options=["Yes", "No", "Defer"],
                timeout_s=120
            )
            ```

        Args:
            prompt: Prompt text shown to the user.
            options: Ordered button labels. The first option maps to approved.
            timeout_s: Continuation deadline in seconds.
            channel: Optional target channel key.
            memory_log_prompt: Enable memory logging for the prompt.
            memory_log_reply: Enable memory logging for a selected choice.
            memory_tags: Optional tags applied to memory entries.

        Returns:
            dict[str, Any]: `{"approved": bool, "choice": Any, "text": str}`.

        Notes:
            If no choice is returned, or `options` is empty, `approved` is `False`.
        """
        channel_key = self._resolve_key(channel)
        button_list = list(options)
        span = await self._tracer.start_span(
            service="channel",
            operation="ask_approval",
            request={
                "prompt": prompt,
                "options": button_list,
                "timeout_s": timeout_s,
                "channel_key": channel_key,
            },
            tags=["channel", "ask", "approval"],
            metadata=self._inject_context_meta({"channel_key": channel_key}),
        )
        try:
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
                payload={"prompt": {"title": prompt, "buttons": button_list}},
                channel=channel,
                timeout_s=timeout_s,
            )
            choice = payload.get("choice")
            text = str(payload.get("text", "") or "")
            if choice is None and text and button_list:
                text_norm = text.strip().lower()
                matched = next(
                    (option for option in button_list if str(option).strip().lower() == text_norm),
                    None,
                )
                if matched is not None:
                    choice = matched
            if choice is not None or text:
                await self._log_chat(
                    "user",
                    f"Selected: {str(choice)}" + (f" | Text: {text}" if text else ""),
                    tags=[*(memory_tags or []), "ask_approval", "reply"],
                    enabled=memory_log_reply,
                    channel=channel,
                )

            if choice is None or not button_list:
                approved = False
            else:
                choice_norm = str(choice).strip().lower()
                first_norm = str(button_list[0]).strip().lower()
                approved = choice_norm == first_norm

            result = {"approved": approved, "choice": choice, "text": text}
            await span.finish(
                response=result,
                metadata=self._inject_context_meta({"channel_key": channel_key}),
            )
            return result
        except Exception as exc:
            await span.fail(exc, metadata=self._inject_context_meta({"channel_key": channel_key}))
            raise

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
        Prompt for file upload and return normalized text/files payload.

        Send a `user_files` continuation prompt, wait for response, and return
        a dictionary with text plus a list-valued `files` field.

        Examples:
            Ask for one or more files:
            ```python
            result = await context.channel().ask_files(
                prompt="Please upload your report."
            )
            # result: { "text": "...", "files": [FileRef(...), ...] }
            ```

            Provide type hints for upload UI:
            ```python
            result = await context.channel().ask_files(
                prompt="Upload images for review.",
                accept=["image/png", ".jpg"],
                multiple=True
            )
            ```

        Args:
            prompt: Prompt text shown with the file picker.
            accept: Optional MIME/extension hints for the adapter UI.
            multiple: Allow multiple selections when `True`.
            timeout_s: Continuation deadline in seconds.
            channel: Optional target channel key.
            memory_log_prompt: Enable memory logging for the prompt.
            memory_log_reply: Enable memory logging for non-empty reply text.
            memory_tags: Optional tags applied to memory entries.

        Returns:
            dict: `{"text": str, "files": list}`.

        Notes:
            `accept` is advisory and adapter-dependent, not strict server-side validation.
        """
        channel_key = self._resolve_key(channel)
        span = await self._tracer.start_span(
            service="channel",
            operation="ask_files",
            request={
                "prompt": prompt,
                "accept": accept or [],
                "multiple": bool(multiple),
                "timeout_s": timeout_s,
                "channel_key": channel_key,
            },
            tags=["channel", "ask", "files"],
            metadata=self._inject_context_meta({"channel_key": channel_key}),
        )
        try:
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

            result = {
                "text": text,
                "files": payload.get("files", [])
                if isinstance(payload.get("files", []), list)
                else [],
            }
            await span.finish(
                response={"text": text, "files_count": len(result["files"])},
                metadata=self._inject_context_meta({"channel_key": channel_key}),
            )
            return result
        except Exception as exc:
            await span.fail(exc, metadata=self._inject_context_meta({"channel_key": channel_key}))
            raise

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
        Prompt for either text or files and return normalized payload.

        Send a `user_input_or_files` continuation request and return both `text`
        and `files` fields after normalization.

        Examples:
            Prompt for either modality:
            ```python
            result = await context.channel().ask_text_or_files(prompt="Reply or upload files")
            ```

            Send to a specific channel:
            ```python
            result = await context.channel().ask_text_or_files(
                prompt="Provide evidence",
                channel="web:chat",
            )
            ```

        Args:
            prompt: Prompt text shown to the user.
            timeout_s: Continuation deadline in seconds.
            channel: Optional target channel key.
            memory_log_prompt: Enable memory logging for the prompt.
            memory_log_reply: Enable memory logging for non-empty reply text.
            memory_tags: Optional tags applied to memory entries.

        Returns:
            dict: `{"text": str, "files": list}`.

        Notes:
            Prefer `ask_text` + `get_latest_uploads` or `ask_files` when modality is known.
        """
        channel_key = self._resolve_key(channel)
        span = await self._tracer.start_span(
            service="channel",
            operation="ask_text_or_files",
            request={"prompt": prompt, "timeout_s": timeout_s, "channel_key": channel_key},
            tags=["channel", "ask", "text_or_files"],
            metadata=self._inject_context_meta({"channel_key": channel_key}),
        )
        try:
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

            result = {
                "text": text,
                "files": payload.get("files", [])
                if isinstance(payload.get("files", []), list)
                else [],
            }
            await span.finish(
                response={"text": text, "files_count": len(result["files"])},
                metadata=self._inject_context_meta({"channel_key": channel_key}),
            )
            return result
        except Exception as exc:
            await span.fail(exc, metadata=self._inject_context_meta({"channel_key": channel_key}))
            raise

    # ---------- inbox helpers (platform-agnostic) ----------
    async def get_latest_uploads(self, *, clear: bool = True) -> list[FileRef]:
        """
        Read latest uploaded files from this channel inbox.

        Load files from the ephemeral KV inbox key for the resolved channel and
        optionally clear the inbox after retrieval.

        Examples:
            Fetch and clear inbox files:
            ```python
            files = await context.channel().get_latest_uploads()
            ```

            Fetch without clearing:
            ```python
            files = await context.channel().get_latest_uploads(clear=False)
            ```

        Args:
            clear: Pop files when `True`; read without removal when `False`.

        Returns:
            list[FileRef]: Uploaded files for this channel inbox.

        Notes:
            This reads inbox uploads regardless of whether they came from `ask_files()`.
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
            # Unique per stream so multiple streams from same node don’t collide
            self._upsert_key = f"{outer._run_id}:{outer._node_id}:stream:{uuid.uuid4().hex}"
            self._phase_group_id = outer._ensure_reply_lifecycle()

        def _inject_context_meta(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            return self._outer._inject_context_meta(meta)

        def _stream_meta(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
            base = dict(extra or {})
            base.setdefault("phase_group_id", self._phase_group_id)
            return self._inject_context_meta(base)

        def _buf(self):
            return getattr(self, "__buf", None)

        def _ensure_buf(self):
            if not hasattr(self, "__buf"):
                self.__buf = []
            return self.__buf

        async def start(self) -> None:
            if not self._started:
                self._started = True
                await self._outer._bus.publish(
                    OutEvent(
                        type="agent.stream.start",
                        channel=self._channel_key,
                        upsert_key=self._upsert_key,
                        meta=self._stream_meta(None),
                    )
                )

        async def delta(self, text_piece: str) -> None:
            """
            Send a text delta for this stream.

            The UI is expected to append `text_piece` to the message associated
            with `upsert_key`. We also keep an internal buffer so `end()` can log
            the full text to memory if needed.
            """
            if not text_piece:
                return

            await self.start()
            buf = self._ensure_buf()
            buf.append(text_piece)

            await self._outer._bus.publish(
                OutEvent(
                    type="agent.stream.delta",
                    channel=self._channel_key,
                    text=text_piece,
                    upsert_key=self._upsert_key,
                    meta=self._stream_meta(None),
                )
            )

        async def end(
            self,
            full_text: str | None = None,
            *,
            memory_log: bool = True,
            memory_role: Literal["assistant", "system", "tool", "user"] = "assistant",
            memory_tags: list[str] | None = None,
            memory_data: dict[str, Any] | None = None,
            memory_severity: int = 2,
            memory_signal: float | None = None,
        ) -> None:
            """
            Finalize the stream.

            - If `full_text` is None, uses the concatenated buffer.
            - Logs the completed text to memory (once).
            - Emits `agent.stream.end` with the final text.
            """
            # Make sure we at least emitted start if no delta was ever sent
            await self.start()

            if full_text is None:
                buf = self._buf()
                full_text = "".join(buf) if buf else ""

            # 1) Memory logging of the completed message
            await self._outer._log_chat(
                memory_role,
                full_text,
                tags=memory_tags,
                data=memory_data,
                severity=memory_severity,
                signal=memory_signal,
                enabled=memory_log,
                channel=self._channel_key,
            )

            # 2) End-of-stream event with final text
            await self._outer._bus.publish(
                OutEvent(
                    type="agent.stream.end",
                    channel=self._channel_key,
                    text=full_text,
                    upsert_key=self._upsert_key,
                    meta=self._stream_meta(None),
                )
            )
            self._outer._close_reply_lifecycle()

    @asynccontextmanager
    async def stream(self, channel: str | None = None) -> AsyncIterator["_StreamSender"]:
        """
        Create a stream sender context for incremental text output.

        Yield a `_StreamSender` bound to the resolved channel. The sender emits
        `agent.stream.start/delta/end` events when `start()`, `delta()`, and
        `end()` are called.

        Examples:
            Stream text deltas:
            ```python
            async with context.channel().stream() as s:
                await s.delta("Hello, ")
                await s.delta("world!")
                await s.end()
            ```

            Stream to a specific channel:
            ```python
            async with context.channel().stream(channel="web:chat") as s:
                await s.delta("Generating results...")
                await s.end(full_text="Results complete.", memory_tags=["llm"])
            ```

        Args:
            channel: Optional target channel key for this stream context.

        Returns:
            AsyncIterator[_StreamSender]: Context yielding the stream sender.

        Notes:
            Caller is responsible for calling `end()`. No auto-end is performed.
        """
        s = ChannelSession._StreamSender(self, channel_key=channel)
        try:
            yield s
        finally:
            # No auto-end; caller decides when to end()
            pass

    async def chat_and_stream(
        self,
        *,
        llm: Any,
        messages: list[dict[str, Any]],
        channel: str | None = None,
        # LLM options
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        reasoning_summary: str | None = None,
        max_output_tokens: int | None = None,
        output_format: str = "text",
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        strict_schema: bool = True,
        validate_json: bool = True,
        fail_on_unsupported: bool = True,
        llm_kwargs: dict[str, Any] | None = None,
        # Thinking phase UX
        thinking_phase: str = "thinking",
        thinking_label_active: str = "Thinking...",
        thinking_label_done: str = "Thinking",
        thinking_detail_interval_s: float = 1.5,
        thinking_detail_tail_chars: int = 300,
        emit_thinking_phase: bool = False,
        # Memory logging
        memory_log: bool = True,
        memory_role: Literal["assistant", "system", "tool", "user"] = "assistant",
        memory_tags: list[str] | None = None,
        memory_data: dict[str, Any] | None = None,
        memory_severity: int = 2,
        memory_signal: float | None = None,
    ) -> tuple[str, dict[str, int], str]:
        """
        Stream an LLM chat response to channel output and return final text/usage.

        This helper wires `llm.chat_stream()` callbacks to channel stream events and
        optional thinking-phase updates (`agent.progress.update`) so agent code does
        not need to reimplement callback plumbing.
        """

        thinking_buffer: list[str] = []
        last_thinking_ts = 0.0
        thinking_started = False
        text_started = False

        async with self.stream(channel=channel) as s:

            async def on_thinking_delta(piece: str) -> None:
                nonlocal last_thinking_ts, thinking_started
                if not emit_thinking_phase:
                    return
                if not piece:
                    return

                thinking_buffer.append(piece)
                now = time.monotonic()

                if not thinking_started:
                    thinking_started = True
                    try:
                        await self.send_phase(
                            phase=thinking_phase,
                            status="active",
                            label=thinking_label_active,
                            channel=channel,
                        )
                    except Exception as e:
                        logger = logging.getLogger("aethergraph.services.channel.session")
                        logger.debug("Failed to send thinking phase update")
                        logger.exception("Error was: %s", str(e))

                if now - last_thinking_ts >= thinking_detail_interval_s:
                    last_thinking_ts = now
                    full = "".join(thinking_buffer)
                    detail = (
                        ("..." + full[-thinking_detail_tail_chars:])
                        if len(full) > thinking_detail_tail_chars
                        else full
                    )
                    try:
                        await self.send_phase(
                            phase=thinking_phase,
                            status="active",
                            label=thinking_label_active,
                            detail=detail,
                            channel=channel,
                        )
                    except Exception as e:
                        logger = logging.getLogger("aethergraph.services.channel.session")
                        logger.debug("Failed to send thinking phase update")
                        logger.exception("Error was: %s", str(e))

            async def on_delta(piece: str) -> None:
                nonlocal text_started
                if not piece:
                    return

                if not text_started and thinking_buffer:
                    text_started = True
                    full = "".join(thinking_buffer)
                    try:
                        await self.send_phase(
                            phase=thinking_phase,
                            status="done",
                            label=thinking_label_done,
                            detail=full,
                            channel=channel,
                        )
                    except Exception as e:
                        logger = logging.getLogger("aethergraph.services.channel.session")
                        logger.debug("Failed to send thinking phase update on completion")
                        logger.exception("Error was: %s", str(e))

                await s.delta(piece)

            kwargs: dict[str, Any] = {
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "thinking_budget": thinking_budget,
                "reasoning_summary": reasoning_summary,
                "max_output_tokens": max_output_tokens,
                "output_format": output_format,
                "json_schema": json_schema,
                "schema_name": schema_name,
                "strict_schema": strict_schema,
                "validate_json": validate_json,
                "fail_on_unsupported": fail_on_unsupported,
                "on_delta": on_delta,
            }
            if emit_thinking_phase:
                kwargs["on_thinking_delta"] = on_thinking_delta
            if llm_kwargs:
                kwargs.update(llm_kwargs)

            resp, usage = await llm.chat_stream(**kwargs)

            if emit_thinking_phase and thinking_buffer and not text_started:
                full = "".join(thinking_buffer)
                try:
                    await self.send_phase(
                        phase=thinking_phase,
                        status="done",
                        label=thinking_label_done,
                        detail=full,
                        channel=channel,
                    )
                except Exception as e:
                    logger = logging.getLogger("aethergraph.services.channel.session")
                    logger.debug("Failed to send thinking phase update on completion")
                    logger.exception("Error was: %s", str(e))

            final_memory_data = dict(memory_data or {})
            if usage:
                final_memory_data.setdefault("usage", usage)
            await s.end(
                full_text=resp,
                memory_log=memory_log,
                memory_role=memory_role,
                memory_tags=memory_tags,
                memory_data=final_memory_data or None,
                memory_severity=memory_severity,
                memory_signal=memory_signal,
            )

        return resp, usage, "".join(thinking_buffer)

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
                            "kind": "progress",
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
                "kind": "progress",
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
                        "kind": "progress",
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
