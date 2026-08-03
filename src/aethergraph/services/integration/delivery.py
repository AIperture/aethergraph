"""Semantic-event emission and exact Channel delivery projection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from aethergraph.contracts.integration import (
    InteractionOption,
    InteractionRequestedPayload,
    MessageCompletedPayload,
    MessageDeltaPayload,
    MessageStartedPayload,
    PhaseChangedPayload,
    ProgressChangedPayload,
    SemanticEvent,
    SemanticEventKind,
    StructuredOutputPayload,
)
from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent

from .events import EventLogSemanticEventStore, PersistedSemanticEvent


class SemanticDeliveryError(RuntimeError):
    """Reject Channel events that cannot enter the semantic delivery contract."""


class SemanticEventEmitter:
    """Allocate durable turn sequences and persist semantic Channel events."""

    def __init__(self, *, deployment_id: str, store: EventLogSemanticEventStore) -> None:
        """Bind semantic emission to one immutable Host deployment.

        Examples:
            Create an emitter:
            ```python
            emitter = SemanticEventEmitter(deployment_id="deployment-1", store=store)
            ```

            Share it across endpoint and provider adapters:
            ```python
            endpoint_adapter = SemanticEventChannelAdapter(emitter=emitter)
            ```

        Args:
            deployment_id: Immutable Host deployment identity.
            store: Canonical semantic event store over the shared EventLog.

        Returns:
            None.

        Notes:
            Sequence allocation initializes from durable session history after restart.
        """
        self.deployment_id = deployment_id
        self.store = store
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._sequences: dict[tuple[str, str], int] = {}

    async def emit(self, event: OutEvent) -> tuple[PersistedSemanticEvent, ...]:
        """Project and persist one exact Channel event in turn order.

        Examples:
            Persist a message:
            ```python
            persisted = await emitter.emit(message_event)
            ```

            Persist an interaction request:
            ```python
            persisted = await emitter.emit(interaction_event)
            ```

        Args:
            event: Exact outbound Channel event with run and session metadata.

        Returns:
            tuple[PersistedSemanticEvent, ...]: Persisted semantic records in order.

        Notes:
            Unsupported event types fail explicitly and are never dropped or converted
            by capability fallback.
        """
        meta = event.meta or {}
        session_id = self._required(meta.get("session_id"), "session_id")
        turn_id = self._required(meta.get("run_id"), "run_id")
        producer = str(meta.get("agent_id") or meta.get("node_id") or "aethergraph")
        drafts = self._project(event)
        key = (session_id, turn_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            sequence = await self._next_sequence(session_id=session_id, turn_id=turn_id)
            persisted: list[PersistedSemanticEvent] = []
            for kind, payload in drafts:
                record = await self.store.append(
                    SemanticEvent(
                        event_id=f"semantic-{uuid4().hex}",
                        deployment_id=self.deployment_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        sequence=sequence,
                        producer=producer,
                        timestamp=datetime.now(UTC),
                        kind=kind,
                        payload=payload,
                        extensions={"aethergraph.channel": event.channel},
                    )
                )
                persisted.append(record)
                sequence += 1
            self._sequences[key] = sequence
        return tuple(persisted)

    async def _next_sequence(self, *, session_id: str, turn_id: str) -> int:
        key = (session_id, turn_id)
        if key in self._sequences:
            return self._sequences[key]
        history = await self.store.list_session(
            deployment_id=self.deployment_id,
            session_id=session_id,
        )
        prior = [item.event.sequence for item in history if item.event.turn_id == turn_id]
        return max(prior, default=-1) + 1

    def _project(self, event: OutEvent) -> tuple[tuple[SemanticEventKind, Any], ...]:
        message_id = event.upsert_key or f"message-{uuid4().hex}"
        if event.type == "agent.stream.start":
            return (
                (SemanticEventKind.MESSAGE_STARTED, MessageStartedPayload(message_id=message_id)),
            )
        if event.type == "agent.stream.delta":
            return (
                (
                    SemanticEventKind.MESSAGE_DELTA,
                    MessageDeltaPayload(message_id=message_id, delta=event.text or ""),
                ),
            )
        if event.type in {"agent.stream.end", "agent.message", "agent.message.update"}:
            artifact_ids = self._artifact_ids(event)
            return (
                (
                    SemanticEventKind.MESSAGE_COMPLETED,
                    MessageCompletedPayload(
                        message_id=message_id,
                        text=event.text or "",
                        artifact_ids=artifact_ids,
                    ),
                ),
            )
        if event.type in {"session.need_input", "session.need_approval"}:
            meta = event.meta or {}
            interaction_id = self._required(meta.get("interaction_id"), "interaction_id")
            options = tuple(
                InteractionOption(
                    option_id=self._required(button.value or button.label, "button value"),
                    label=button.label,
                )
                for button in (event.buttons or [])
            )
            kind = str(meta.get("interaction_kind") or "user_input")
            request_kind = {
                "approval": "approval",
                "choice": "choice",
                "user_files": "files",
                "user_input_or_files": "files",
                "user_input": "text",
            }.get(kind)
            if request_kind is None:
                raise SemanticDeliveryError(f"Unsupported interaction kind: {kind}")
            return (
                (
                    SemanticEventKind.INTERACTION_REQUESTED,
                    InteractionRequestedPayload(
                        interaction_id=interaction_id,
                        request_kind=request_kind,
                        prompt=event.text or "",
                        options=options,
                        allow_multiple=bool(meta.get("multiple", False)),
                        accepted_content_types=tuple(meta.get("accept") or ()),
                    ),
                ),
            )
        if event.type in {
            "agent.progress.start",
            "agent.progress.update",
            "agent.progress.end",
        }:
            rich = event.rich or {}
            if rich.get("kind") == "phase":
                return (
                    (
                        SemanticEventKind.PHASE_CHANGED,
                        PhaseChangedPayload(
                            phase=self._required(rich.get("phase"), "phase"),
                            status=str(rich.get("status") or "active"),
                            label=str(rich.get("label") or rich.get("phase")),
                            detail=str(rich.get("detail")) if rich.get("detail") else None,
                        ),
                    ),
                )
            status = {
                "agent.progress.start": "started",
                "agent.progress.update": "running",
                "agent.progress.end": "completed" if rich.get("success", True) else "failed",
            }[event.type]
            return (
                (
                    SemanticEventKind.PROGRESS_CHANGED,
                    ProgressChangedPayload(
                        progress_id=event.upsert_key
                        or self._required(rich.get("progress_id"), "progress_id"),
                        status=status,
                        label=str(rich.get("label") or rich.get("title") or "Progress"),
                        current=rich.get("current"),
                        total=rich.get("total"),
                        unit=rich.get("unit"),
                        detail=rich.get("detail") or rich.get("subtitle"),
                    ),
                ),
            )
        if event.type in {"file.upload", "link.buttons", "session.waiting"}:
            return (
                (
                    SemanticEventKind.STRUCTURED_OUTPUT,
                    StructuredOutputPayload(
                        output_name=f"channel.{event.type}",
                        value={
                            "text": event.text,
                            "buttons": [
                                {"label": button.label, "value": button.value, "url": button.url}
                                for button in (event.buttons or [])
                            ],
                            "file": self._json_file(event.file),
                        },
                    ),
                ),
            )
        raise SemanticDeliveryError(f"Unsupported Channel event type: {event.type}")

    @staticmethod
    def _artifact_ids(event: OutEvent) -> tuple[str, ...]:
        value = (event.file or {}).get("artifact_id") or (event.file or {}).get("id")
        return (str(value),) if value else ()

    @staticmethod
    def _json_file(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            key: item
            for key, item in value.items()
            if isinstance(item, str | int | float | bool) or item is None
        }

    @staticmethod
    def _required(value: Any, name: str) -> str:
        if value is None or value == "":
            raise SemanticDeliveryError(f"Semantic delivery requires {name}.")
        return str(value)


class SemanticEventChannelAdapter:
    """Persist semantic history before exact optional transport projection."""

    def __init__(
        self,
        *,
        emitter: SemanticEventEmitter,
        downstream: ChannelAdapter | None = None,
    ) -> None:
        """Create a persistence-only or persistence-and-delivery adapter.

        Examples:
            Create an Agent Endpoint adapter:
            ```python
            adapter = SemanticEventChannelAdapter(emitter=emitter)
            ```

            Wrap a Slack projector:
            ```python
            adapter = SemanticEventChannelAdapter(emitter=emitter, downstream=slack)
            ```

        Args:
            emitter: Manifest-bound semantic event emitter.
            downstream: Explicit provider projector, or `None` for endpoint history only.

        Returns:
            None.

        Notes:
            Persistence failure prevents downstream delivery, preserving ordered history.
        """
        self.emitter = emitter
        self.downstream = downstream
        self.capabilities = (
            set(getattr(downstream, "capabilities", set()))
            if downstream is not None
            else {"text", "buttons", "file", "stream", "edit"}
        )

    async def send(self, event: OutEvent) -> dict | None:
        """Persist one semantic projection and deliver through the exact adapter.

        Examples:
            Persist an endpoint event:
            ```python
            await adapter.send(event)
            ```

            Persist and send a provider event:
            ```python
            result = await provider_adapter.send(event)
            ```

        Args:
            event: Exact outbound Channel event.

        Returns:
            dict | None: Downstream delivery metadata, or semantic cursor metadata
            for a persistence-only endpoint adapter.

        Notes:
            There is no alternate delivery path if persistence or projection fails.
        """
        persisted = await self.emitter.emit(event)
        if self.downstream is not None:
            return await self.downstream.send(event)
        return {"event_cursor": persisted[-1].cursor if persisted else None}
