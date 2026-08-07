from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal

from aethergraph.contracts.services.memory import (
    EXTERNAL_RESOURCE_CHANGED_KIND,
    Event,
    ExternalResourceChangedEvent,
)
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.storage.vector_index.utils import build_index_meta_from_scope

from .utils import normalize_tags, now_iso, stable_event_id

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


class WriteMixin:
    async def record_raw(
        self: MemoryFacadeProtocol,
        *,
        base: dict[str, Any],
        text: str | None = None,
        metrics: dict[str, float] | None = None,
        state_key: str | None = None,
        expected_state_revision: int | None = None,
    ) -> Event:
        """Record a low-level event from an explicit `base` payload.

        Intro:
            This primitive fills scope identity, timestamps, a stable Event id,
            and signal metadata before persistence and optional search indexing.
            State helpers may request a durable conditional append.

        Examples:
            Record an ordinary event:
            ```python
            event = await memory.record_raw(
                base={"kind": "checkpoint", "data": {"step": 2}},
                text="checkpoint two",
            )
            ```

            Record a conditional state snapshot:
            ```python
            event = await memory.record_raw(
                base={"kind": "state.snapshot", "data": snapshot_payload},
                state_key="agent:writer",
                expected_state_revision=3,
            )
            ```

        Args:
            base: Raw Event fields including kind, tags, data, and identity overrides.
            text: Human-readable content used for previews and search indexing.
            metrics: Optional numeric metrics attached to the event.
            state_key: Exact state key for a conditional snapshot append.
            expected_state_revision: Exact current durable snapshot revision.

        Returns:
            Event: The persisted event, including its generated ``event_id``.

        Notes:
            Conditional snapshots reach durable persistence before the hot log,
            so rejected writers cannot leave a visible hot-only Event.
        """
        if (state_key is None) != (expected_state_revision is None):
            raise ValueError("state_key and expected_state_revision must be supplied together")
        span = await self._start_trace(
            operation="record_raw",
            request={"base": base, "text": text, "metrics": metrics},
            tags=["memory", "record"],
            metrics=metrics,
        )
        try:
            ts_iso = now_iso()
            ts_num = time.time()
            dims: dict[str, str] = {}
            if self.scope is not None:
                dims = self.scope.identity_labels()
            run_id = base.get("run_id") or dims.get("run_id") or self.run_id
            session_id = base.get("session_id") or dims.get("session_id") or self.session_id
            scope_id = base.get("scope_id") or self.memory_scope_id or session_id or run_id
            user_id = base.get("user_id") or dims.get("user_id")
            org_id = base.get("org_id") or dims.get("org_id")
            client_id = base.get("client_id") or dims.get("client_id")
            graph_id = base.get("graph_id") or dims.get("graph_id") or self.graph_id
            node_id = base.get("node_id") or dims.get("node_id") or self.node_id
            app_id = base.get("app_id") or dims.get("app_id")
            agent_id = base.get("agent_id") or dims.get("agent_id")
            base["tags"] = normalize_tags(base.get("tags"))
            severity = int(base.get("severity", 2))
            signal = base.get("signal")
            if signal is None:
                signal = self._estimate_signal(text=text, metrics=metrics, severity=severity)
            kind = base.get("kind") or "misc"
            eid = str(base.get("event_id") or "").strip() or stable_event_id(
                {
                    "ts": ts_iso,
                    "run_id": run_id,
                    "kind": kind,
                    "text": (text or "")[:6000],
                    "tool": base.get("tool"),
                    "topic": base.get("topic"),
                }
            )
            evt = Event(
                event_id=eid,
                ts=ts_iso,
                run_id=run_id,
                scope_id=scope_id,
                user_id=user_id,
                org_id=org_id,
                client_id=client_id,
                session_id=session_id,
                kind=kind,
                stage=base.get("stage"),
                text=text,
                tags=base.get("tags"),
                data=base.get("data"),
                metrics=metrics,
                graph_id=graph_id,
                node_id=node_id,
                app_id=app_id,
                agent_id=agent_id,
                tool=base.get("tool"),
                topic=base.get("topic"),
                severity=severity,
                signal=float(signal or 0.0),
                inputs=base.get("inputs"),
                outputs=base.get("outputs"),
                embedding=base.get("embedding"),
                pii_flags=base.get("pii_flags"),
                version=2,
            )
            if expected_state_revision is None:
                await self.hotlog.append(
                    self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit
                )
                await self.persistence.append_event(self.timeline_id, evt)
            else:
                await self.persistence.append_state_snapshot_if_revision(
                    self.timeline_id,
                    evt,
                    state_key=str(state_key),
                    expected_revision=int(expected_state_revision),
                )
                await self.hotlog.append(
                    self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit
                )
            if self.scoped_indices is not None and self.scoped_indices.backend is not None:
                try:
                    preview = (text or "")[:500] if text else ""
                    meta = build_index_meta_from_scope(
                        kind=str(evt.kind),
                        source="memory",
                        ts=ts_iso,
                        created_at_ts=ts_num,
                        extra={
                            "run_id": evt.run_id,
                            "scope_id": evt.scope_id,
                            "session_id": evt.session_id,
                            "app_id": evt.app_id,
                            "agent_id": evt.agent_id,
                            "graph_id": evt.graph_id,
                            "node_id": evt.node_id,
                            "stage": evt.stage,
                            "tags": evt.tags or [],
                            "severity": evt.severity,
                            "signal": evt.signal,
                            "tool": evt.tool,
                            "topic": evt.topic,
                            "timeline_id": self.timeline_id,
                            "client_id": evt.client_id,
                            "user_id": evt.user_id,
                            "org_id": evt.org_id,
                            "preview": preview,
                        },
                    )
                    await self.scoped_indices.upsert(
                        corpus="event",
                        item_id=evt.event_id,
                        text=evt.text or "",
                        metadata=meta,
                    )
                except Exception:
                    if self.logger:
                        self.logger.exception("Error indexing memory event %s", evt.event_id)
            try:
                meter = current_metering()
                await meter.record_event(scope=self.scope, scope_id=scope_id, kind=f"memory.{kind}")
            except Exception:
                if self.logger:
                    self.logger.exception("Error recording metering event")
            await span.finish(
                response={"event_id": evt.event_id, "kind": evt.kind},
                metadata=self._trace_meta({"event_id_ref": evt.event_id}),
                metrics=metrics,
            )
            return evt
        except Exception as exc:
            await span.fail(exc, metadata=self._trace_meta(), metrics=metrics)
            raise

    async def append_event(
        self: MemoryFacadeProtocol,
        *,
        kind: str,
        data: Any,
        tags: list[str] | None = None,
        severity: int = 2,
        stage: str | None = None,
        inputs=None,
        outputs=None,
        metrics: dict[str, float] | None = None,
        signal: float | None = None,
        text: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
    ) -> Event:
        """Record a structured event with automatic text/data normalization.

        Serializes ``data`` into an indexable ``text`` preview when ``text`` is not
        given, truncates overly long text, and wraps non-dict payloads so they are
        always JSON-safe. Delegates persistence to :meth:`record_raw`. This is the
        general-purpose recording entry point most callers should use.

        Args:
            kind: Logical event type (e.g. ``"chat.turn"``, ``"tool_result"``).
            data: Arbitrary JSON-serializable payload for the event.
            tags: Low-cardinality labels used for filtering and search.
            severity: Importance level (1=low, 2=medium, 3=high).
            stage: Optional phase indicator (e.g. role for chat turns).
            inputs: Optional structured input values.
            outputs: Optional structured output values.
            metrics: Optional numeric metrics.
            signal: Optional relevance signal; estimated automatically when omitted.
            text: Optional human-readable content; derived from ``data`` when omitted.
            topic: Optional topic classification.
            tool: Optional tool topic associated with the event.

        Returns:
            Event: The persisted event.
        """
        if text is None and data is not None:
            if isinstance(data, str):
                text = data
            else:
                try:
                    text = json.dumps(data, ensure_ascii=False)
                except Exception as exc:
                    text = f"<unserializable data: {exc!s}>"
        if text and len(text) > 2000:
            text = text[:2000] + " ...[truncated]"
        data_field: dict[str, Any] | None = None
        if isinstance(data, dict):
            data_field = data
        elif data is not None and not isinstance(data, str):
            try:
                json.dumps(data, ensure_ascii=False)
                data_field = {"value": data}
            except Exception:
                data_field = {"repr": repr(data)}
        base = {
            "kind": kind,
            "stage": stage,
            "severity": severity,
            "tags": normalize_tags(tags),
            "data": data_field,
            "inputs": inputs,
            "outputs": outputs,
            "topic": topic,
            "tool": tool,
        }
        if signal is not None:
            base["signal"] = signal
        return await self.record_raw(base=base, text=text, metrics=metrics)

    async def append_external_resource_change(
        self: MemoryFacadeProtocol,
        change: ExternalResourceChangedEvent | dict[str, Any],
    ) -> Event:
        """Ingest one committed authoritative-store outbox event.

        The method validates scope identity and persists the compact change on
        the existing memory timeline. It does not mutate the resource and has no
        run submission, continuation, notification, or scheduling behavior.

        Examples:
            Ingest a typed event:
            ```python
                change = ExternalResourceChangedEvent(
                    event_id="evt-1",
                    scope_id=memory.memory_scope_id,
                    session_id=memory.session_id,
                    source_sequence=1,
                    resource_key="design_config:project-42",
                    resource_kind="design_config",
                    revision="19",
                    source="design_ui",
                    recorded_at="2026-07-10T20:00:01Z",
                )
                persisted = await memory.append_external_resource_change(change)
                assert persisted.event_id == "evt-1"
            ```

            Ingest a JSON outbox row:
            ```python
                persisted = await memory.append_external_resource_change(
                    {
                        "event_id": "evt-2",
                        "scope_id": memory.memory_scope_id,
                        "session_id": memory.session_id,
                        "source_sequence": 2,
                        "resource_key": "clock:world",
                        "resource_kind": "clock",
                        "revision": "day-2",
                        "source": "world_service",
                        "recorded_at": "2026-07-10T20:00:02Z",
                    }
                )
                assert persisted.kind == "external.resource.changed"
            ```

        Args:
            change: Typed contract or strict JSON row read from a committed
                producer outbox.

        Returns:
            Event: Persisted memory event retaining the producer event identity.

        Notes:
            Producers must deliver rows in `source_sequence` order within one
            source and scope. Consumers use that sequence rather than ingestion
            timestamps as the durable cursor.
        """

        normalized = (
            change
            if isinstance(change, ExternalResourceChangedEvent)
            else ExternalResourceChangedEvent.from_dict(change)
        )
        if normalized.scope_id != str(self.memory_scope_id or ""):
            raise ValueError("external resource event scope_id does not match memory scope")
        if normalized.session_id != str(self.session_id or ""):
            raise ValueError("external resource event session_id does not match memory session")
        payload = normalized.to_dict()
        text = normalized.summary or (
            f"external resource changed: {normalized.resource_key} "
            f"revision {normalized.revision}"
        )
        return await self.record_raw(
            base={
                "event_id": normalized.event_id,
                "kind": EXTERNAL_RESOURCE_CHANGED_KIND,
                "session_id": normalized.session_id,
                "scope_id": normalized.scope_id,
                "stage": "committed_outbox",
                "severity": 2,
                "tags": [
                    "external_resource",
                    EXTERNAL_RESOURCE_CHANGED_KIND,
                    f"external_source:{normalized.source}",
                    f"resource_kind:{normalized.resource_kind}",
                ],
                "data": payload,
                "topic": normalized.resource_key,
            },
            text=text,
        )

    async def append_chat_turn(
        self: MemoryFacadeProtocol,
        role: Literal["user", "assistant", "system", "tool"],
        text: str,
        *,
        tags: list[str] | None = None,
        data: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
    ) -> Event:
        """Record a single chat turn as a ``chat.turn`` event.

        Stores the message ``role`` as the event ``stage`` and tags the event with
        ``"chat"`` so it can be retrieved by :meth:`recent_chat` and
        :meth:`chat_history_for_llm`.

        Args:
            role: Message role (``"user"``, ``"assistant"``, ``"system"``, ``"tool"``).
            text: The message content.
            tags: Extra tags to attach (``"chat"`` is always added).
            data: Optional extra fields merged into the event payload.
            severity: Importance level (1=low, 2=medium, 3=high).
            signal: Optional relevance signal.

        Returns:
            Event: The persisted chat event.
        """
        payload = {"role": role, "text": text}
        if data:
            payload.update(data)
        return await self.append_event(
            kind="chat.turn",
            data=payload,
            tags=["chat", *normalize_tags(tags)],
            severity=severity,
            stage=role,
            signal=signal,
            text=text,
        )

    async def append_tool_result(
        self: MemoryFacadeProtocol,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """Record the result of a tool invocation as a ``tool_result`` event.

        The ``tool`` name is stored as both the event ``tool`` and ``topic`` so tool
        history can be filtered by tool (e.g.
        ``query_events(kinds=["tool_result"], tool=...)``).

        Args:
            tool: Tool name/identifier.
            inputs: Structured inputs passed to the tool.
            outputs: Structured outputs returned by the tool.
            tags: Extra tags to attach.
            metrics: Optional numeric metrics (e.g. latency, cost).
            message: Optional human-readable summary used for search/preview.
            severity: Importance level (defaults to 3).

        Returns:
            Event: The persisted tool-result event.
        """
        return await self.append_event(
            kind="tool_result",
            data={"tool": tool},
            tags=tags,
            severity=severity,
            inputs=inputs or [],
            outputs=outputs or [],
            metrics=metrics,
            text=message,
            tool=tool,
            topic=tool,
        )

    async def append_state_snapshot(
        self: MemoryFacadeProtocol,
        key: str,
        value: Any,
        *,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        severity: int = 2,
        signal: float | None = None,
        kind: str = "state.snapshot",
        stage: str | None = None,
        expected_revision: int | None = None,
    ) -> Event:
        """Persist a JSON-serializable value as a revisioned state Event.

        Intro:
            Dataclasses and model objects become plain data. Supplying an
            expected revision makes the durable comparison and append atomic.

        Examples:
            Append an ordinary snapshot:
            ```python
            event = await memory.append_state_snapshot("agent:writer", state)
            ```

            Append revision four only after revision three:
            ```python
            event = await memory.append_state_snapshot(
                "agent:writer",
                state,
                expected_revision=3,
            )
            ```

        Args:
            key: Logical state key this snapshot belongs to.
            value: The state value (any JSON-serializable / dataclass / model object).
            tags: Extra tags to attach.
            meta: Optional metadata stored alongside the value.
            severity: Importance level (1=low, 2=medium, 3=high).
            signal: Optional relevance signal.
            kind: Event kind to use (defaults to ``"state.snapshot"``).
            stage: Optional phase indicator.
            expected_revision: Optional exact current durable enclosing revision.

        Returns:
            Event: The persisted state event.

        Notes:
            Conditional snapshots carry revision `expected_revision + 1` and
            raise `StateSnapshotConflictError` without changing the hot log.
        """
        import dataclasses

        def _to_serializable(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return _to_serializable(dataclasses.asdict(obj))
            if hasattr(obj, "model_dump"):
                try:
                    return _to_serializable(obj.model_dump())
                except Exception:
                    pass
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {str(k): _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_to_serializable(v) for v in obj]
            return {"__repr__": repr(obj)}

        snapshot_meta = dict(meta or {})
        if expected_revision is not None:
            if isinstance(expected_revision, bool) or int(expected_revision) < 0:
                raise ValueError("expected_revision must be a non-negative integer")
            next_revision = int(expected_revision) + 1
            authored_revision = snapshot_meta.get("revision")
            if authored_revision is not None and int(authored_revision) != next_revision:
                raise ValueError("snapshot metadata revision must equal expected_revision + 1")
            snapshot_meta["revision"] = next_revision
        payload = {"key": key, "value": _to_serializable(value), "meta": snapshot_meta}
        index_text = f"state:{key} "
        try:
            index_text += json.dumps(payload["value"], ensure_ascii=False, sort_keys=True)
        except Exception:
            index_text += repr(payload["value"])
        event_tags = ["state", f"state:{key}", *normalize_tags(tags)]
        if expected_revision is None:
            return await self.append_event(
                kind=kind,
                data=payload,
                tags=event_tags,
                severity=severity,
                stage=stage,
                signal=signal,
                text=index_text,
            )
        return await self.record_raw(
            base={
                "kind": kind,
                "data": payload,
                "tags": event_tags,
                "severity": severity,
                "stage": stage,
                "signal": signal,
            },
            text=index_text,
            state_key=key,
            expected_state_revision=int(expected_revision),
        )
