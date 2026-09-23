from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from aethergraph.contracts.services.trigger import TriggerKind, TriggerService
from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.observability.canonical_service import CanonicalObservationService
from aethergraph.observability.models import ObservationRecord, ObservationScope
from aethergraph.services.scope.scope import Scope
from aethergraph.storage.contracts import StorageConflictError

from .scheduling import _initial_fire_at, _validate_trigger_config
from .types import TriggerRecord


@dataclass
class TriggerServiceImpl(TriggerService):
    """Create and mutate triggers through tenant-bound service operations."""

    store: TriggerStore
    observation_sink: CanonicalObservationService | None = None
    logger: Any | None = None

    async def create_from_scope(
        self,
        *,
        scope: Scope,
        graph_id: str,
        default_inputs: dict[str, Any],
        kind: TriggerKind,
        cron_expr: str | None = None,
        interval_seconds: int | None = None,
        run_at: datetime | None = None,
        event_key: str | None = None,
        tz: str | None = None,
        max_overlap_runs: int | None = None,
        catch_up_missed: bool = False,
        origin: str = "schedule",
        trigger_name: str | None = None,
        meta: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> TriggerRecord:
        """Create one validated trigger or recover its exact prior creation receipt.

        A scoped idempotency key reuses the existing schedule even after firing or
        cancellation. Reuse with different creation parameters fails; retries never
        reset the current cursor, revision, activity state or next firing time.

        Examples:
            Create a periodic wakeup:
            ```python
            trigger = await service.create_from_scope(scope=scope, graph_id="wake",
                default_inputs={"agent": "main"}, kind="interval", interval_seconds=60,
                idempotency_key="agent-main-wakeup")
            ```

            Recover the same creation after an acknowledgement was lost:
            ```python
            trigger = await service.create_from_scope(scope=scope, graph_id="wake",
                default_inputs={"agent": "main"}, kind="interval", interval_seconds=60,
                idempotency_key="agent-main-wakeup")
            ```

        Args:
            scope: Tenant, session and Agent owning the schedule.
            graph_id: Registered execution target.
            default_inputs: Serializable graph inputs.
            kind: Cron, interval, one-shot or event schedule.
            cron_expr: Expression for cron schedules.
            interval_seconds: Interval recurrence period.
            run_at: Exact time for a one-shot schedule.
            event_key: Matching event identity for event schedules.
            tz: Optional schedule time zone.
            max_overlap_runs: Maximum overlapping executions.
            catch_up_missed: Whether restart should recover missed occurrences.
            origin: Run origin attached to scheduled execution.
            trigger_name: Optional display label.
            meta: Caller metadata, excluding service-owned creation evidence.
            idempotency_key: Optional 1..512 character identity within the owning scope.

        Returns:
            TriggerRecord: Created record or the current state of the identical prior creation.

        Notes:
            Run and node identities do not partition creation keys. A new independent
            schedule needs a new key. Omitting the key creates a new schedule each time.
        """
        _validate_trigger_config(
            kind=kind,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            run_at=run_at,
            event_key=event_key,
            tz=tz,
            max_overlap_runs=max_overlap_runs,
        )
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 512
        ):
            raise ValueError("Trigger idempotency_key must contain 1..512 characters")
        if "_creation_digest" in (meta or {}):
            raise ValueError("Trigger metadata _creation_digest is service-owned")
        trigger_id = f"trig-{uuid4().hex[:8]}"
        if idempotency_key is not None:
            identity = json.dumps(
                [
                    scope.org_id,
                    scope.user_id,
                    scope.client_id,
                    scope.session_id,
                    scope.agent_id,
                    idempotency_key,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            trigger_id = f"trig-{sha256(identity.encode('utf-8')).hexdigest()}"
        now = datetime.now(UTC)
        trig = TriggerRecord.from_scope(
            trigger_id=trigger_id,
            scope=scope,
            graph_id=graph_id,
            default_inputs=default_inputs,
            kind=kind,
            origin=origin,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            run_at=run_at,
            event_key=event_key,
            tz=tz,
            max_overlap_runs=max_overlap_runs,
            catch_up_missed=catch_up_missed,
            meta=meta,
            trigger_name=trigger_name,
        )
        if idempotency_key is not None:
            request = trig.to_dict()
            for field in ("created_at", "active", "last_fired_at", "next_fire_at"):
                request.pop(field)
            digest = sha256(
                json.dumps(
                    request,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            trig.meta = {**trig.meta, "_creation_digest": digest}
            previous = await self.store.get(trigger_id)
            if previous is not None:
                return self._validate_creation_replay(previous, digest)
        trig.next_fire_at = _initial_fire_at(trig, now)
        try:
            await self.store.create(trig)
        except StorageConflictError:
            # Another creator may have won after the read, or the first creation
            # was committed before its acknowledgement was lost. Never update or
            # reactivate the existing schedule while recovering that receipt.
            if idempotency_key is None:
                raise
            previous = await self.store.get(trigger_id)
            if previous is None:
                raise
            return self._validate_creation_replay(previous, digest)
        await self._log_trigger_event(trig, action="created")
        return trig

    @staticmethod
    def _validate_creation_replay(previous: TriggerRecord, digest: str) -> TriggerRecord:
        if previous.meta.get("_creation_digest") != digest:
            raise StorageConflictError(
                "Trigger idempotency key was reused with a different request"
            )
        return previous

    async def cancel(
        self,
        trigger_id: str,
        *,
        org_id: str | None,
        user_id: str | None,
        client_id: str | None,
    ) -> bool:
        """Deactivate one trigger only when it belongs to the supplied owner."""
        trig = await self._get_owned(
            trigger_id, org_id=org_id, user_id=user_id, client_id=client_id
        )
        if trig is None:
            return False
        trig.active = False
        trig.next_fire_at = None
        await self.store.update(trig)
        await self._log_trigger_event(trig, action="canceled")
        return True

    async def delete(
        self,
        trigger_id: str,
        *,
        org_id: str | None,
        user_id: str | None,
        client_id: str | None,
    ) -> bool:
        """Delete one trigger only when it belongs to the supplied owner."""
        trig = await self._get_owned(
            trigger_id, org_id=org_id, user_id=user_id, client_id=client_id
        )
        if trig is None:
            return False
        await self.store.delete(trigger_id)
        await self._log_trigger_event(trig, action="deleted")
        return True

    async def get(
        self,
        trigger_id: str,
        *,
        org_id: str | None,
        user_id: str | None,
        client_id: str | None,
    ) -> TriggerRecord | None:
        """Read one trigger only when it belongs to the supplied owner."""
        return await self._get_owned(
            trigger_id, org_id=org_id, user_id=user_id, client_id=client_id
        )

    async def list_for_owner(
        self, *, org_id: str | None, user_id: str | None
    ) -> list[TriggerRecord]:
        """List triggers visible to one normalized tenant owner."""
        if org_id is None and user_id is None:
            return []
        return await self.store.list_all(org_id=org_id, user_id=user_id)

    async def _get_owned(
        self,
        trigger_id: str,
        *,
        org_id: str | None,
        user_id: str | None,
        client_id: str | None,
    ) -> TriggerRecord | None:
        if org_id is None and user_id is None and client_id is None:
            return None
        trig = await self.store.get(trigger_id)
        if trig is None:
            return None
        if org_id is not None and trig.org_id != org_id:
            return None
        if user_id is not None and trig.user_id != user_id and trig.client_id != user_id:
            return None
        if client_id is not None and trig.client_id != client_id:
            return None
        return trig

    async def _log_trigger_event(self, trig: TriggerRecord, action: str) -> None:
        if self.observation_sink is None:
            return
        try:
            await self.observation_sink.append_observation(
                ObservationRecord(
                    observation_id=f"trig-evt-{uuid4().hex[:8]}",
                    category="trigger",
                    name=action,
                    summary=f"Trigger {action}",
                    scope=_trigger_observation_scope(trig),
                    attributes={
                        "action": action,
                        "trigger_id": trig.trigger_id,
                        "kind": trig.kind,
                        "graph_id": trig.graph_id,
                        "meta": trig.meta or {},
                    },
                )
            )
        except Exception:
            if self.logger:
                self.logger.exception("Failed to log trigger event for %s", trig.trigger_id)


def _trigger_observation_scope(
    trig: TriggerRecord,
    *,
    run_id: str | None = None,
) -> ObservationScope:
    return ObservationScope(
        org_id=trig.org_id,
        user_id=trig.user_id,
        app_id=trig.app_id,
        session_id=trig.session_id,
        run_id=run_id,
        agent_id=trig.agent_id,
        graph_id=trig.graph_id,
    )
