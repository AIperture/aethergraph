from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.contracts.services.trigger import TriggerKind
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.scope.scope import ScopeLevel
from aethergraph.services.triggers.trigger_service import TriggerService
from aethergraph.services.triggers.types import TriggerRecord

router = APIRouter(prefix="/triggers", tags=["triggers"])


class TriggerCreateRequest(BaseModel):
    graph_id: str = Field(..., description="ID of the graph to trigger")
    kind: TriggerKind = Field(..., description="cron | interval | event | one_shot")
    trigger_name: str | None = Field(
        None, description="Optional human-friendly name for the trigger"
    )

    # What to pass to the graph
    default_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Default inputs to the graph; can be overridden by event payload for event-based triggers",
    )

    # Trigger configuration (mutually exclusive based on kind)
    cron_expr: str | None = Field(
        default=None,
        description="Cron expression for cron triggers (e.g. '0 9 * * *' for every day at 9am)",
    )
    interval_seconds: int | None = Field(
        default=None, description="Interval in seconds for interval triggers", ge=1
    )
    run_at: datetime | None = Field(
        default=None, description="Exact time to run for one_shot triggers"
    )
    event_key: str | None = Field(
        default=None, description="Event key to listen for event triggers"
    )
    tz: str | None = Field(
        default=None,
        description="Timezone for cron triggers (e.g. 'America/Los_Angeles'); defaults to UTC if not set",
    )

    # Behavior knobs
    max_overlap_runs: int | None = Field(
        default=None,
        description="If set, max number of overlapping runs allowed; excess runs will be skipped",
        ge=0,
    )
    catch_up_missed: bool = Field(
        default=False,
        description="If true, missed runs (e.g. due to downtime) will be triggered on startup; if false, they will be skipped",
    )

    # Optional scopng hints for memory/agent/session
    memory_level: ScopeLevel | None = Field(
        default=None,
        description="Optional hint for the memory level to use for runs spawned by this trigger; if not set, inherits from the creating scope",
    )
    agent_id: str | None = Field(
        default=None,
        description="Optional agent_id to associate with runs spawned by this trigger; if not set, inherits from the creating scope",
    )
    app_id: str | None = Field(
        default=None,
        description="Optional app_id to associate with runs spawned by this trigger; if not set, inherits from the creating scope",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session_id to associate with runs spawned by this trigger; if not set, inherits from the creating scope",
    )

    # Freeform metadata
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Optional freeform metadata to store with the trigger"
    )


class FireEventRequest(BaseModel):
    event_key: str = Field(..., description="Event key to fire")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Optional payload to include with the event"
    )


class TriggerMeta(BaseModel):
    trigger_id: str
    trigger_name: str | None

    # Ownership / identity
    org_id: str | None
    user_id: str | None
    client_id: str | None
    app_id: str | None
    agent_id: str | None
    session_id: str | None
    memory_level: ScopeLevel | None

    # What + how
    graph_id: str | None
    kind: TriggerKind
    cron_expr: str | None
    interval_seconds: int | None
    run_at: datetime | None
    event_key: str | None
    tz: str | None
    max_overlap_runs: int | None
    catch_up_missed: bool

    # Lifecycle
    active: bool
    created_at: datetime
    last_fired_at: datetime | None
    next_fire_at: datetime | None

    # UI-only status string (derived from active + next_fire_at for convenience
    status: str

    # Freeform metadata
    meta: dict[str, Any]


class TriggerListResponse(BaseModel):
    triggers: list[TriggerMeta]


def _trigger_status(rec: TriggerRecord) -> str:
    """
    Simple, UI-friendly status string for a trigger, derived from its active flag and next_fire_at time.
    """
    if not rec.active:
        # One shot that has fired or been canceled
        if rec.kind == "one_shot" and rec.last_fired_at is not None:
            return "finished"
        return "inactive"

    # Active triggers
    if rec.kind == "event":
        return "listening"
    if rec.next_fire_at is None:
        return "pending"
    return "scheduled"


def _trigger_to_meta(rec: TriggerRecord) -> TriggerMeta:
    return TriggerMeta(
        trigger_id=rec.trigger_id,
        trigger_name=rec.trigger_name,
        org_id=rec.org_id,
        user_id=rec.user_id,
        client_id=rec.client_id,
        app_id=rec.app_id,
        agent_id=rec.agent_id,
        session_id=rec.session_id,
        memory_level=rec.memory_level,
        graph_id=rec.graph_id,
        kind=rec.kind,
        cron_expr=rec.cron_expr,
        interval_seconds=rec.interval_seconds,
        run_at=rec.run_at,
        event_key=rec.event_key,
        tz=rec.tz,
        max_overlap_runs=rec.max_overlap_runs,
        catch_up_missed=rec.catch_up_missed,
        active=rec.active,
        created_at=rec.created_at,
        last_fired_at=rec.last_fired_at,
        next_fire_at=rec.next_fire_at,
        status=_trigger_status(rec),
        meta=rec.meta or {},
    )


def _tenant_for_identity(identity: RequestIdentity) -> dict[str, str | None]:
    """
    Normalization: Treat 'user' as user_id OR client_id.
    """
    user_or_client = identity.user_id or identity.client_id
    return {
        "org_id": identity.org_id,
        "user_id": user_or_client,
        "client_id": identity.client_id,
    }


# just for double-checking in the engine that a trigger belongs to the firing identity;
# we should be filtering at the store level but this is a sanity check to avoid rogue triggers slipping through
def _check_trigger_belongs_to_identity(
    rec: TriggerRecord,
    identity: RequestIdentity,
) -> bool:
    t = _tenant_for_identity(identity)

    if t["org_id"] is not None and rec.org_id != t["org_id"]:
        return False

    if t["user_id"] is not None:  # noqa: SIM102
        if not (rec.user_id == t["user_id"] or rec.client_id == t["user_id"]):
            return False

    if t["client_id"] is not None and rec.client_id != t["client_id"]:  # noqa: SIM103
        return False

    return True


@router.get("", response_model=TriggerListResponse)
async def list_triggers(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TriggerListResponse:
    """
    List all triggers owned by the caller's identity (org + user/client).
    """
    services = current_services()
    trigger_svc: TriggerService = services.trigger_service
    tenant = _tenant_for_identity(identity)
    recs = await trigger_svc.list_for_owner(**tenant)

    metas = [_trigger_to_meta(rec) for rec in recs]
    return TriggerListResponse(triggers=metas)


@router.post("", response_model=TriggerMeta, status_code=status.HTTP_201_CREATED)
async def create_trigger(
    payload: TriggerCreateRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TriggerMeta:
    """
    Create a new trigger based on the provided configuration.
    """
    services = current_services()
    trigger_svc: TriggerService = services.trigger_service
    scope_factory = services.scope_factory

    # Build a scope representing "this user/app/client/org" for the trigger; this is used for scoping triggers in the store and later for scoping memory/agent/session when the trigger fires
    # TODO: check if we need to use the memory_level/agent_id/app_id/session_id hints from the request to further customize the scope
    scope = scope_factory.for_trigger(identity=identity)

    if payload.kind == "cron" and not payload.cron_expr:
        raise HTTPException(status_code=400, detail="cron_expr is required for cron triggers")
    if payload.kind == "interval" and not payload.interval_seconds:
        raise HTTPException(
            status_code=400, detail="interval_seconds is required for interval triggers"
        )
    if payload.kind == "one_shot" and not payload.run_at:
        raise HTTPException(status_code=400, detail="run_at is required for one_shot triggers")
    if payload.kind == "event" and not payload.event_key:
        raise HTTPException(status_code=400, detail="event_key is required for event triggers")

    rec = await trigger_svc.create_from_scope(
        scope=scope,
        graph_id=payload.graph_id,
        default_inputs=payload.default_inputs or {},
        kind=payload.kind,
        cron_expr=payload.cron_expr,
        interval_seconds=payload.interval_seconds,
        run_at=payload.run_at,
        event_key=payload.event_key,
        tz=payload.tz,
        max_overlap_runs=payload.max_overlap_runs,
        catch_up_missed=payload.catch_up_missed,
        origin="schedule",
        trigger_name=payload.trigger_name,
        meta=payload.meta or {},
    )

    return _trigger_to_meta(rec)


@router.get("/{trigger_id}", response_model=TriggerMeta)
async def get_trigger(
    trigger_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TriggerMeta:
    """
    Get details of a specific trigger by ID, if it belongs to the caller's identity.
    """
    services = current_services()
    trigger_svc: TriggerService = services.trigger_service
    rec = await trigger_svc.get(trigger_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Check ownership
    if not _check_trigger_belongs_to_identity(rec, identity):
        raise HTTPException(status_code=404, detail="Trigger not found")

    return _trigger_to_meta(rec)


@router.delete("/{trigger_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trigger(
    trigger_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> None:
    """
    Cancel (deactivate) a trigger by ID, if it belongs to the caller's identity.
    """
    services = current_services()
    trigger_svc: TriggerService = services.trigger_service
    rec = await trigger_svc.get(trigger_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Check ownership
    if not _check_trigger_belongs_to_identity(rec, identity):
        raise HTTPException(status_code=404, detail="Trigger not found")

    await trigger_svc.cancel(trigger_id)


@router.delete("/{trigger_id}/hard", status_code=status.HTTP_204_NO_CONTENT)
async def hard_delete_trigger(
    trigger_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> None:
    """
    Hard delete a trigger by ID, if it belongs to the caller's identity. This is irreversible.
    """
    services = current_services()
    trigger_svc: TriggerService = services.trigger_service
    rec = await trigger_svc.get(trigger_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Check ownership
    if not _check_trigger_belongs_to_identity(rec, identity):
        raise HTTPException(status_code=404, detail="Trigger not found")

    await trigger_svc.delete(trigger_id)


@router.post("/fire-event", status_code=status.HTTP_204_NO_CONTENT)
async def fire_event(
    payload: FireEventRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> None:
    """
    Fire an event-based trigger by event_key, with optional payload. This will execute all active triggers matching the event_key and caller's identity (org + user/client).
    """
    services = current_services()
    trigger_engine = services.trigger_engine
    await trigger_engine.fire_event(
        event_key=payload.event_key,
        payload=payload.payload,
        org_id=identity.org_id,
        user_id=identity.user_id,
        client_id=identity.client_id,
    )
    return {"ok": True}


@router.post("/fire-event-global", status_code=status.HTTP_204_NO_CONTENT)
async def fire_event_global(
    payload: FireEventRequest,
) -> None:
    """
    Fire an event-based trigger by event_key, with optional payload, without any tenant scoping. This will execute all active triggers matching the event_key, regardless of their org/user/client association. Use with caution.
    """
    services = current_services()
    trigger_engine = services.trigger_engine
    await trigger_engine.fire_event(
        event_key=payload.event_key,
        payload=payload.payload,
        org_id=None,
        user_id=None,
        client_id=None,
    )
    return {"ok": True}
