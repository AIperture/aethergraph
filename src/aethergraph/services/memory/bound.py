from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from aethergraph.contracts.services.memory import Event
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.memory.resolver import ResolverContext, resolve_params
Value = Dict[str, Any]

@dataclass
class BoundMemory:
    mem: MemoryFacade
    defaults: Dict[str, Any] = None  # run_id, graph_id, node_id, agent_id (usually injected by NodeContext)

    async def record(self, *, kind: str, text: Optional[str] = None, severity: int = 2,
                     stage: Optional[str] = None, tags: Optional[List[str]] = None,
                     entities: Optional[List[str]] = None, metrics: Optional[Dict[str, Any]] = None,
                     inputs_ref: Optional[Dict[str, Any]] = None, outputs_ref: Optional[Dict[str, Any]] = None,
                     sources: Optional[List[str]] = None, signal: float | None = None) -> Event:
        base = {
            **(self.defaults or {}),
            "kind": kind,
            "stage": stage,
            "severity": severity,
            "tags": tags or [],
            "entities": entities or [],
            "inputs_ref": inputs_ref,
            "outputs_ref": outputs_ref,
            "sources": sources,
            "signal": float(signal if signal is not None else self._estimate_signal(text=text, metrics=metrics, severity=severity)),
        }
        return await self.mem.record_raw(base=base, text=text, metrics=metrics)

    async def user(self, text: str):         return await self.record(kind="user_msg", text=text, stage="observe")
    async def assistant(self, text: str):    return await self.record(kind="assistant_msg", text=text, stage="act")
    async def tool_start(self, note=None):   return await self.record(kind="tool_start", text=note, stage="act")
    async def tool_ok(self, note=None, metrics=None): return await self.record(kind="tool_result", text=note, metrics=metrics, stage="observe", severity=3)
    async def tool_error(self, err: Exception): return await self.record(kind="error", text=f"{type(err).__name__}: {err}", severity=5, stage="observe")

    async def write_result(self, *, topic: str, inputs: List[Value] | None = None,
                           outputs: List[Value] | None = None, tags: List[str] | None = None,
                           metrics: Dict[str, float] | None = None, message: str | None = None,
                           severity: int = 3) -> Event:
        return await self.mem.write_result(
            topic=topic, inputs=inputs or [], outputs=outputs or [],
            tags=tags, metrics=metrics, message=message, severity=severity
        )

    async def resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rctx = ResolverContext(mem=self.mem)
        return await resolve_params(params, rctx)

    # ---- helper ----
    def _estimate_signal(self, *, text: Optional[str], metrics: Optional[Dict[str, Any]], severity:int) -> float:
        score = 0.15 + 0.1 * severity
        if text:    score += min(len(text) / 400.0, 0.4)
        if metrics: score += 0.2
        return max(0.0, min(1.0, score))
