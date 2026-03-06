# memory-related inspection

from contextlib import suppress
from datetime import datetime, timezone
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query  # type: ignore

from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.contracts.services.memory import Event, MemoryTenantFilter
from aethergraph.core.runtime.run_types import RunRecord
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.memory.facade.core import derive_timeline_id
from aethergraph.services.memory.storage_filters import event_time
from aethergraph.services.scope.tenant import registry_tenant_from_identity

from .deps import RequestIdentity, get_identity
from .schemas.memory import (
    MemoryEvent,
    MemoryEventListResponse,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySummaryEntry,
    MemorySummaryListResponse,
)

router = APIRouter(tags=["memory"])


def _parse_ts(ts: str | float | int) -> datetime:
    if isinstance(ts, int | float):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _snippet_from_event(evt: Event, max_len: int = 120) -> str:
    raw: str | None = None
    if evt.text:
        raw = evt.text
    elif isinstance(evt.data, dict):
        data_text = evt.data.get("text")
        if isinstance(data_text, str) and data_text.strip():
            raw = data_text
        else:
            with suppress(Exception):
                raw = json.dumps(evt.data, ensure_ascii=False, sort_keys=True)

    snippet = " ".join(str(raw or "").split())
    if len(snippet) <= max_len:
        return snippet
    return snippet[: max_len - 1].rstrip() + "..."


def _tenant_filter_for_identity(identity: RequestIdentity) -> MemoryTenantFilter | None:
    tenant: MemoryTenantFilter = {}
    if identity.org_id:
        tenant["org_id"] = identity.org_id
    if identity.user_id:
        tenant["user_id"] = identity.user_id
    if identity.client_id:
        tenant["client_id"] = identity.client_id
    return tenant or None


def _resolve_memory_config_for_run(
    container: Any,
    record: RunRecord,
    identity: RequestIdentity,
) -> tuple[str, str | None]:
    registry = getattr(container, "registry", None)
    tenant = registry_tenant_from_identity(identity) if identity is not None else None
    level = "session" if record.agent_id else "run"
    custom_scope_id: str | None = None
    meta: dict[str, Any] = {}

    if registry is not None:
        if record.agent_id:
            meta = (
                registry.get_meta(
                    nspace="agent",
                    name=record.agent_id,
                    version=None,
                    tenant=tenant,
                    include_global=True,
                )
                or {}
            )
        elif record.app_id:
            meta = (
                registry.get_meta(
                    nspace="app",
                    name=record.app_id,
                    version=None,
                    tenant=tenant,
                    include_global=True,
                )
                or {}
            )
        elif record.graph_id:
            meta = (
                registry.get_meta(
                    "graphfn", record.graph_id, None, tenant=tenant, include_global=True
                )
                or registry.get_meta(
                    "graph", record.graph_id, None, tenant=tenant, include_global=True
                )
                or {}
            )

    if "memory" in meta:
        level = meta["memory"].get("level", level)
        custom_scope_id = meta["memory"].get("scope")

    return level, custom_scope_id


def _timeline_id_for_run(container: Any, record: RunRecord, identity: RequestIdentity) -> str:
    scope_factory = getattr(container, "scope_factory", None)
    if scope_factory is None:
        return record.session_id or record.run_id

    level, custom_scope_id = _resolve_memory_config_for_run(container, record, identity)
    mem_scope = scope_factory.for_memory(
        identity=identity,
        run_id=record.run_id,
        graph_id=record.graph_id,
        session_id=record.session_id,
        app_id=record.app_id,
        agent_id=record.agent_id,
        level=level,
        custom_scope_id=custom_scope_id,
    )
    return derive_timeline_id(
        memory_scope_id=mem_scope.memory_scope_id(),
        run_id=record.run_id,
        org_id=mem_scope.org_id,
    )


async def _resolve_timeline_ids(
    *,
    container: Any,
    identity: RequestIdentity,
    scope_id: str | None,
    session_id: str | None,
    run_id: str | None,
    agent_id: str | None,
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | None) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)

    if scope_id:
        return [scope_id]

    rm = getattr(container, "run_manager", None)
    candidate_records: list[RunRecord] = []

    if run_id and rm is not None:
        record = await rm.get_record(run_id)
        if record is not None:
            candidate_records.append(record)

    if session_id and rm is not None:
        records = await rm.list_records(session_id=session_id, limit=200, offset=0)
        records.sort(key=lambda rec: rec.started_at, reverse=True)
        if agent_id:
            records = [rec for rec in records if rec.agent_id == agent_id]
        if run_id:
            records = [rec for rec in records if rec.run_id == run_id]
        candidate_records.extend(records)

    for record in candidate_records:
        add(_timeline_id_for_run(container, record, identity))

    if resolved:
        return resolved

    if session_id:
        scope_factory = getattr(container, "scope_factory", None)
        if scope_factory is not None:
            fallback_scope = scope_factory.for_memory(
                identity=identity,
                run_id=run_id or session_id,
                session_id=session_id,
                agent_id=agent_id,
                level="session",
                custom_scope_id=None,
            )
            add(
                derive_timeline_id(
                    memory_scope_id=fallback_scope.memory_scope_id(),
                    run_id=run_id or session_id,
                    org_id=fallback_scope.org_id,
                )
            )
            return resolved

    if run_id:
        return [run_id]

    return []


def _event_to_api_event(evt: Event) -> MemoryEvent:
    created_at = _parse_ts(evt.ts)
    data: dict[str, Any] | None = None
    if evt.data is not None:
        data = evt.data
    elif evt.text:
        data = {"text": evt.text}

    return MemoryEvent(
        event_id=evt.event_id,
        scope_id=evt.scope_id or evt.run_id,
        ts=evt.ts,
        session_id=evt.session_id,
        agent_id=evt.agent_id,
        run_id=evt.run_id,
        node_id=evt.node_id,
        graph_id=evt.graph_id,
        kind=evt.kind,
        stage=evt.stage,
        topic=evt.topic,
        tool=evt.tool,
        tags=evt.tags or [],
        severity=evt.severity,
        signal=evt.signal,
        created_at=created_at,
        snippet=_snippet_from_event(evt),
        text=evt.text,
        data=data or {},
        metrics=evt.metrics,
        inputs=evt.inputs,
        outputs=evt.outputs,
    )


def _doc_to_summary_entry(doc_id: str, doc: dict[str, Any]) -> MemorySummaryEntry:
    ts_str = doc.get("ts") or doc.get("created_at") or ""
    created_at = _parse_ts(ts_str) if ts_str else datetime.utcnow()
    tw = doc.get("time_window") or {}
    from_str = tw.get("from") or tw.get("start") or ""
    to_str = tw.get("to") or tw.get("end") or ""
    time_from = _parse_ts(from_str) if from_str else created_at
    time_to = _parse_ts(to_str) if to_str else created_at
    text = doc.get("summary") or doc.get("text") or ""
    meta_keys = {"summary", "text", "scope_id", "run_id", "summary_tag", "ts", "time_window"}
    metadata = {k: v for k, v in doc.items() if k not in meta_keys}
    return MemorySummaryEntry(
        summary_id=doc_id,
        scope_id=doc.get("scope_id") or doc.get("run_id") or "",
        summary_tag=doc.get("summary_tag"),
        created_at=created_at,
        time_from=time_from,
        time_to=time_to,
        text=text,
        metadata=metadata,
    )


def _string_score(haystack: str, needle: str) -> float:
    if not needle:
        return 0.0
    return 1.0 if needle.lower() in haystack.lower() else 0.0


@router.get("/memory/events", response_model=MemoryEventListResponse)
async def list_memory_events(
    scope_id: Annotated[str | None, Query(description="Memory timeline / scope id")] = None,  # noqa: B008
    session_id: Annotated[
        str | None, Query(description="Session boundary for debug memory")
    ] = None,  # noqa: B008
    agent_id: Annotated[str | None, Query(description="Filter to a specific agent")] = None,  # noqa: B008
    run_id: Annotated[str | None, Query(description="Filter to a specific run")] = None,  # noqa: B008
    kinds: Annotated[
        str | None, Query(description="Comma-separated list of kinds to filter")
    ] = None,  # noqa: B008
    tags: Annotated[str | None, Query(description="Comma-separated list of tags to filter")] = None,  # noqa: B008
    after: Annotated[datetime | None, Query()] = None,  # noqa: B008
    before: Annotated[datetime | None, Query()] = None,  # noqa: B008
    cursor: Annotated[str | None, Query()] = None,  # noqa: B008
    limit: Annotated[int, Query(ge=1, le=50)] = 20,  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemoryEventListResponse:
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        return MemoryEventListResponse(events=[], next_cursor=None)

    timeline_ids = await _resolve_timeline_ids(
        container=container,
        identity=identity,
        scope_id=scope_id,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
    )

    tenant = _tenant_filter_for_identity(identity)
    kinds_list = _parse_csv(kinds)
    tags_list = _parse_csv(tags)

    raw_events: list[Event] = []
    for timeline_id in timeline_ids:
        raw_events.extend(
            await mem_factory.persistence.query_events(
                timeline_id,
                tenant=tenant,
                since=after.isoformat() if after else None,
                until=before.isoformat() if before else None,
                kinds=kinds_list,
                tags=tags_list,
                agent_id=agent_id,
                limit=None,
                offset=0,
            )
        )

    deduped: list[Event] = []
    seen_event_ids: set[str] = set()
    for evt in raw_events:
        if evt.event_id in seen_event_ids:
            continue
        seen_event_ids.add(evt.event_id)
        deduped.append(evt)

    deduped.sort(key=lambda e: (event_time(e), e.event_id), reverse=True)
    offset = decode_cursor(cursor)
    page = deduped[offset : offset + limit]
    next_cursor = encode_cursor(offset + limit) if len(deduped) > offset + limit else None
    return MemoryEventListResponse(
        events=[_event_to_api_event(e) for e in page],
        next_cursor=next_cursor,
    )


@router.get("/memory/summaries", response_model=MemorySummaryListResponse)
async def list_memory_summaries(
    scope_id: Annotated[str, Query()],
    summary_tag: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySummaryListResponse:
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        return MemorySummaryListResponse(summaries=[], next_cursor=None)

    offset = decode_cursor(cursor)
    docs = await mem_factory.persistence.query_summaries(
        scope_id=scope_id,
        tenant=_tenant_filter_for_identity(identity),
        summary_tag=summary_tag,
        limit=limit,
        offset=offset,
    )
    entries = [
        _doc_to_summary_entry(
            str(doc.get("summary_doc_id") or doc.get("doc_id") or f"summary-{idx}"),
            doc,
        )
        for idx, doc in enumerate(docs)
    ]
    entries.sort(key=lambda e: e.created_at, reverse=True)
    next_cursor = encode_cursor(offset + limit) if len(entries) == limit else None
    return MemorySummaryListResponse(summaries=entries, next_cursor=next_cursor)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    req: MemorySearchRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySearchResponse:
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        return MemorySearchResponse(hits=[])

    timeline_id = req.scope_id or ""
    query = req.query or ""
    top_k = getattr(req, "top_k", 10) or 10
    tenant = _tenant_filter_for_identity(identity)
    hits: list[MemorySearchHit] = []

    if timeline_id:
        raw_events: list[Event] = await mem_factory.hotlog.query(
            timeline_id,
            tenant=tenant,
            kinds=None,
            limit=mem_factory.hot_limit,
        )
        for evt in raw_events:
            text_parts: list[str] = []
            if evt.text:
                text_parts.append(evt.text)
            if evt.data:
                with suppress(Exception):
                    text_parts.append(str(evt.data))
            score = _string_score(" ".join(text_parts), query)
            if score <= 0.0:
                continue
            hits.append(MemorySearchHit(score=score, event=_event_to_api_event(evt), summary=None))

    summaries = await mem_factory.persistence.query_summaries(
        timeline_id=timeline_id or None,
        tenant=tenant,
        summary_tag=getattr(req, "summary_tag", None),
        limit=top_k * 5,
        offset=0,
    )
    for idx, doc in enumerate(summaries):
        text_parts: list[str] = []
        if doc.get("summary"):
            text_parts.append(str(doc.get("summary")))
        if doc.get("key_facts"):
            with suppress(Exception):
                text_parts.append(" ".join(map(str, doc["key_facts"])))
        score = _string_score(" ".join(text_parts), query)
        if score <= 0.0:
            continue
        doc_id = str(doc.get("summary_doc_id") or doc.get("doc_id") or f"summary-{idx}")
        hits.append(
            MemorySearchHit(
                score=score,
                event=None,
                summary=_doc_to_summary_entry(doc_id, doc),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    if len(hits) > top_k:
        hits = hits[:top_k]
    return MemorySearchResponse(hits=hits)
