from __future__ import annotations
import hmac
from typing import Any, Dict, Optional
from jsonschema import validate, ValidationError
from logging import getLogger
from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.resume import ResumeBus

log = getLogger(__name__)

class ResumeRouter:
    def __init__(self, *, store: AsyncContinuationStore, runner: ResumeBus, logger=None, wait_registry=None):
        self.store = store
        self.runner = runner
        self.logger = logger or log
        self.waits = wait_registry

    async def resume(self, run_id: str, node_id: str, token: str, payload: Dict[str, Any]) -> None:
        cont = await self.store.get(run_id, node_id)
        if not cont:
            self.logger.error("No continuation for %s/%s", run_id, node_id)
            raise PermissionError("Invalid continuation or token")

        if not hmac.compare_digest(token, cont.token):
            self.logger.error("Invalid token for %s/%s", run_id, node_id)
            raise PermissionError("Invalid continuation or token")
        
        # Merge continuation payload (setup-time) with incoming resume payload (adapter-time)
        base_payload = getattr(cont, "payload", None) or {}
        full_payload: Dict[str, Any] = {**base_payload, **(payload or {})}

        # Cooperative fast path
        if self.waits and token in getattr(self.waits, "_futs", {}):
            try:
                self.waits.resolve(token, full_payload)
                self.logger.info("Resolved cooperative wait for %s/%s", run_id, node_id)
                try:
                    await self.store.delete(run_id, node_id)
                except Exception as e:
                    self.logger.warning("Failed to delete continuation after cooperative resolution: %s", e)
                return
            except Exception as e:
                self.logger.error("Error resolving cooperative wait for %s/%s: %s", run_id, node_id, e, exc_info=True)
                raise

        # Schema validate
        if cont.resume_schema:
            try:
                validate(instance=payload, schema=cont.resume_schema)  # validate incoming payload only
            except ValidationError as e:
                self.logger.error("Resume payload validation error: %s", e.message)
                raise ValueError(f"Invalid resume payload: {e.message}")

        # Hand off to scheduler bus
        await self.runner.enqueue_resume(run_id=run_id, node_id=node_id, token=token, payload=full_payload)
