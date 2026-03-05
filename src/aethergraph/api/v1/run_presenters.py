from __future__ import annotations

from typing import Any

from aethergraph.api.v1.schemas.runs import RunSummary


def extract_app_id_from_tags(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("client:") or tag.startswith("flow:"):
            continue
        return tag
    return None


def registry_graph_meta(reg: Any, *, kind: str | None, graph_id: str) -> tuple[str | None, bool]:
    if reg is None:
        return (None, False)

    if kind == "taskgraph":
        meta = reg.get_meta(nspace="graph", name=graph_id, version=None) or {}
    elif kind == "graphfn":
        meta = reg.get_meta(nspace="graphfn", name=graph_id, version=None) or {}
    else:
        meta = {}
    return (meta.get("flow_id"), bool(meta.get("entrypoint", False)))


def to_run_summary(rec: Any, *, reg: Any, flow_id_override: str | None = None) -> RunSummary:
    flow_id, entrypoint = registry_graph_meta(reg, kind=rec.kind, graph_id=rec.graph_id)
    effective_flow_id = flow_id_override or rec.meta.get("flow_id") or flow_id
    app_id = rec.app_id or rec.meta.get("app_id") or extract_app_id_from_tags(rec.tags)

    return RunSummary(
        run_id=rec.run_id,
        graph_id=rec.graph_id,
        status=rec.status,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        tags=rec.tags,
        user_id=rec.user_id,
        org_id=rec.org_id,
        session_id=rec.session_id or None,
        graph_kind=rec.kind,
        flow_id=effective_flow_id,
        entrypoint=entrypoint,
        meta=rec.meta or {},
        app_id=app_id,
        app_name=rec.meta.get("app_name"),
        agent_id=rec.agent_id or rec.meta.get("agent_id") or None,
        origin=rec.origin,
        visibility=rec.visibility,
        importance=rec.importance,
        artifact_count=rec.get("artifact_count"),
        last_artifact_at=rec.get("last_artifact_at"),
    )
