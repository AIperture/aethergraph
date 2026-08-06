"""Semantic-event emission and exact Channel delivery projection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
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
    ToolActivityPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
)
from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.core.runtime.run_types import RunStatus

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
                        extensions={
                            "aethergraph.channel": event.channel,
                            **(
                                {"aethergraph.upsert_key": event.upsert_key}
                                if event.upsert_key
                                else {}
                            ),
                        },
                    )
                )
                persisted.append(record)
                sequence += 1
            self._sequences[key] = sequence
        return tuple(persisted)

    async def emit_semantic(
        self,
        *,
        session_id: str,
        turn_id: str,
        producer: str,
        kind: SemanticEventKind,
        payload: Any,
        extensions: dict[str, Any] | None = None,
    ) -> PersistedSemanticEvent:
        """Persist one Host-authored semantic event in canonical turn order.

        This boundary is for lifecycle facts that do not originate as Channel
        presentation events, such as the terminal state of a submitted run.

        Examples:
            Record successful completion:
            ```python
            await emitter.emit_semantic(
                session_id="session-1",
                turn_id="run-1",
                producer="aethergraph.run_manager",
                kind=SemanticEventKind.TURN_COMPLETED,
                payload=TurnCompletedPayload(result_available=True),
            )
            ```

            Record terminal failure:
            ```python
            await emitter.emit_semantic(
                session_id="session-1",
                turn_id="run-1",
                producer="aethergraph.run_manager",
                kind=SemanticEventKind.TURN_FAILED,
                payload=TurnFailedPayload(
                    code="run_failed",
                    message="Execution failed.",
                    retryable=False,
                ),
            )
            ```

        Args:
            session_id: Exact AG session identity.
            turn_id: Exact submitted root run identity.
            producer: Stable producer identity for the lifecycle fact.
            kind: Canonical semantic event kind.
            payload: Payload matching the selected semantic event kind.
            extensions: Optional namespaced host metadata.

        Returns:
            PersistedSemanticEvent: Durable record with its allocated cursor.

        Notes:
            Sequence allocation shares the same per-turn lock and durable history as
            Channel-originated semantic events.
        """
        normalized_session_id = self._required(session_id, "session_id")
        normalized_turn_id = self._required(turn_id, "turn_id")
        normalized_producer = self._required(producer, "producer")
        key = (normalized_session_id, normalized_turn_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            sequence = await self._next_sequence(
                session_id=normalized_session_id,
                turn_id=normalized_turn_id,
            )
            record = await self.store.append(
                SemanticEvent(
                    event_id=f"semantic-{uuid4().hex}",
                    deployment_id=self.deployment_id,
                    session_id=normalized_session_id,
                    turn_id=normalized_turn_id,
                    sequence=sequence,
                    producer=normalized_producer,
                    timestamp=datetime.now(UTC),
                    kind=kind,
                    payload=payload,
                    extensions=dict(extensions or {}),
                )
            )
            self._sequences[key] = sequence + 1
        return record

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
            if event.rich is not None:
                return (
                    (
                        SemanticEventKind.STRUCTURED_OUTPUT,
                        StructuredOutputPayload(
                            output_name="channel.rich",
                            value={"text": event.text, "rich": event.rich},
                        ),
                    ),
                )
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
        if event.type == "structured.output":
            rich = event.rich or {}
            return (
                (
                    SemanticEventKind.STRUCTURED_OUTPUT,
                    StructuredOutputPayload(
                        output_name=self._required(rich.get("output_name"), "output_name"),
                        value=rich.get("value"),
                    ),
                ),
            )
        if event.type == "agent.tool.activity":
            rich = event.rich or {}
            return (
                (
                    SemanticEventKind.TOOL_ACTIVITY,
                    ToolActivityPayload(
                        tool_call_id=self._required(rich.get("tool_call_id"), "tool_call_id"),
                        tool_name=self._required(rich.get("tool_name"), "tool_name"),
                        status=self._required(rich.get("status"), "status"),
                        message=(
                            str(rich.get("message")) if rich.get("message") is not None else None
                        ),
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


class SemanticTurnMonitor:
    """Publish terminal semantic state for submitted integration turns."""

    def __init__(self, *, run_manager: Any, emitter: SemanticEventEmitter) -> None:
        """Bind terminal observation to one RunManager and semantic emitter.

        Examples:
            Create the Host monitor:
            ```python
            monitor = SemanticTurnMonitor(run_manager=manager, emitter=emitter)
            ```

            Pass it to the root dispatcher:
            ```python
            dispatcher = AGRootTurnDispatcher(container, turn_monitor=monitor)
            ```

        Args:
            run_manager: Host RunManager providing durable terminal records.
            emitter: Canonical deployment-bound semantic event emitter.

        Returns:
            None.

        Notes:
            Observation tasks are retained until completion so they are not lost to
            garbage collection while a run is active or waiting for continuation.
        """
        self.run_manager = run_manager
        self.emitter = emitter
        self._tasks: set[asyncio.Task[None]] = set()

    def observe(
        self,
        *,
        run_id: str,
        session_id: str,
        route_id: str,
        integration_id: str,
    ) -> None:
        """Begin terminal observation for one submitted root turn.

        Examples:
            Observe an endpoint turn:
            ```python
            monitor.observe(
                run_id="run-1",
                session_id="session-1",
                route_id="studio-assistant",
                integration_id="agstudio",
            )
            ```

            Observe a provider turn:
            ```python
            monitor.observe(
                run_id="run-2",
                session_id="session-2",
                route_id="support",
                integration_id="slack-main",
            )
            ```

        Args:
            run_id: Exact submitted root run identity.
            session_id: Bound AG session identity.
            route_id: Immutable manifest route identity.
            integration_id: Immutable integration identity.

        Returns:
            None: Observation has been scheduled on the active event loop.

        Notes:
            A waiting run remains observed until it succeeds, fails, or is canceled.
        """
        task = asyncio.create_task(
            self._observe(
                run_id=run_id,
                session_id=session_id,
                route_id=route_id,
                integration_id=integration_id,
            ),
            name=f"semantic-turn:{run_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._finish_task)

    def _finish_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logging.getLogger("aethergraph.integration.semantic_turn").exception(
                "Terminal semantic event emission failed",
                exc_info=error,
            )

    async def _observe(
        self,
        *,
        run_id: str,
        session_id: str,
        route_id: str,
        integration_id: str,
    ) -> None:
        record = await self.run_manager.wait_run(run_id)
        extensions = {
            "aethergraph.integration_id": integration_id,
            "aethergraph.route_id": route_id,
        }
        if record.status == RunStatus.succeeded:
            await self.emitter.emit_semantic(
                session_id=session_id,
                turn_id=run_id,
                producer="aethergraph.run_manager",
                kind=SemanticEventKind.TURN_COMPLETED,
                payload=TurnCompletedPayload(result_available=bool(record.result_available)),
                extensions=extensions,
            )
            return
        code = "run_canceled" if record.status == RunStatus.canceled else "run_failed"
        message = str(record.error or code.replace("_", " ").capitalize())[:4_000]
        await self.emitter.emit_semantic(
            session_id=session_id,
            turn_id=run_id,
            producer="aethergraph.run_manager",
            kind=SemanticEventKind.TURN_FAILED,
            payload=TurnFailedPayload(code=code, message=message, retryable=False),
            extensions=extensions,
        )
