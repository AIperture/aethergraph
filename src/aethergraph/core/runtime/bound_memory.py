from typing import Any, Dict, List, Optional
from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.memory.io_helpers import Value 

# TODO: Deprecate this adapter in favor of direct MemoryFacade usage in runtime contexts.

class BoundMemoryAdapter:
    """Minimal adapter to preserve ctx.mem().* API while delegating to MemoryFacade."""
    def __init__(self, mem: MemoryFacade, defaults: Dict[str, Any]):
        self._mem = mem
        self._defaults = defaults

    async def record(self, *, kind: str, text: Optional[str] = None, severity: int = 2,
                     stage: Optional[str] = None, tags: Optional[List[str]] = None,
                     entities: Optional[List[str]] = None, metrics: Optional[Dict[str, Any]] = None,
                     inputs_ref: Optional[Dict[str, Any]] = None, outputs_ref: Optional[Dict[str, Any]] = None,
                     sources: Optional[List[str]] = None, signal: Optional[float] = None):
        base = dict(
            **self._defaults,
            kind=kind, stage=stage, severity=severity,
            tags=tags or [], entities=entities or [],
            inputs_ref=inputs_ref, outputs_ref=outputs_ref,
            signal=signal,
        )
        return await self._mem.record_raw(base=base, text=text, metrics=metrics, sources=sources)

    async def user(self, text: str):      return await self.record(kind="user_msg", text=text, stage="observe")
    async def assistant(self, text: str): return await self.record(kind="assistant_msg", text=text, stage="act")

    async def write_result(self, *, topic: str, inputs: Optional[List[Value]] = None,
                           outputs: Optional[List[Value]] = None, tags: Optional[List[str]] = None,
                           metrics: Optional[Dict[str, float]] = None, message: Optional[str] = None,
                           severity: int = 3):
        return await self._mem.write_result(
            topic=topic, inputs=inputs or [], outputs=outputs or [],
            tags=tags, metrics=metrics, message=message, severity=severity
        )
