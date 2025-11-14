from typing import Any

from aethergraph.services.memory.facade import MemoryFacade
from aethergraph.services.memory.io_helpers import Value

# TODO: Deprecate this adapter in favor of direct MemoryFacade usage in runtime contexts.


class BoundMemoryAdapter:
    """Minimal adapter to preserve ctx.mem().* API while delegating to MemoryFacade."""

    def __init__(self, mem: MemoryFacade, defaults: dict[str, Any]):
        self._mem = mem
        self._defaults = defaults

    async def record(
        self,
        *,
        kind: str,
        text: str | None = None,
        severity: int = 2,
        stage: str | None = None,
        tags: list[str] | None = None,
        entities: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        inputs_ref: dict[str, Any] | None = None,
        outputs_ref: dict[str, Any] | None = None,
        sources: list[str] | None = None,
        signal: float | None = None,
    ):
        base = dict(
            **self._defaults,
            kind=kind,
            stage=stage,
            severity=severity,
            tags=tags or [],
            entities=entities or [],
            inputs_ref=inputs_ref,
            outputs_ref=outputs_ref,
            signal=signal,
        )
        return await self._mem.record_raw(base=base, text=text, metrics=metrics, sources=sources)

    async def user(self, text: str):
        return await self.record(kind="user_msg", text=text, stage="observe")

    async def assistant(self, text: str):
        return await self.record(kind="assistant_msg", text=text, stage="act")

    async def write_result(
        self,
        *,
        topic: str,
        inputs: list[Value] | None = None,
        outputs: list[Value] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ):
        return await self._mem.write_result(
            topic=topic,
            inputs=inputs or [],
            outputs=outputs or [],
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )
