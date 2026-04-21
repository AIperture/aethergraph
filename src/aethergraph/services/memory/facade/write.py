from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal

from aethergraph.contracts.services.memory import Event
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
    ) -> Event:
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
            eid = stable_event_id(
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
            await self.hotlog.append(
                self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit
            )
            await self.persistence.append_event(self.timeline_id, evt)
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
    ) -> Event:
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

        payload = {"key": key, "value": _to_serializable(value), "meta": meta or {}}
        index_text = f"state:{key} "
        try:
            index_text += json.dumps(payload["value"], ensure_ascii=False, sort_keys=True)
        except Exception:
            index_text += repr(payload["value"])
        return await self.append_event(
            kind=kind,
            data=payload,
            tags=["state", f"state:{key}", *normalize_tags(tags)],
            severity=severity,
            stage=stage,
            signal=signal,
            text=index_text,
        )
