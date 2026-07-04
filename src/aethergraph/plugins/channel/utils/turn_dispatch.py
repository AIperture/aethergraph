from __future__ import annotations

from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.services.channel.resources import InputResource, InputResourceNormalizer


def resources_from_file_refs(
    files: list[dict[str, Any]] | None,
    *,
    source: str,
) -> list[InputResource]:
    normalizer = InputResourceNormalizer()
    out: list[InputResource] = []
    for file_info in files or []:
        raw = dict(file_info)
        raw["source"] = source
        raw["meta"] = raw.get("meta") or raw.get("extra") or {}
        out.append(normalizer.from_dict(raw, source=source))
    return out


async def dispatch_channel_turn_run(
    *,
    container: Any,
    identity: RequestIdentity | None,
    agent_id: str,
    text: str,
    attachments: list[InputResource] | None,
    session_id: str | None = None,
    user_meta: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    origin: RunOrigin = RunOrigin.chat,
) -> str:
    identity = identity or RequestIdentity(user_id="local", org_id="local", mode="local")
    registry = scoped_registry(identity)
    agent_meta = registry.get_meta(nspace="agent", name=agent_id)
    if not agent_meta:
        raise ValueError(f"Agent not found: {agent_id}")

    backing = agent_meta.get("backing", {})
    if backing.get("type") != "graphfn":
        raise ValueError(
            f"Unsupported agent backing type: {backing.get('type')}. Only 'graphfn' is supported."
        )

    graph_id = backing["name"]
    run_vis = RunVisibility(agent_meta.get("run_visibility", RunVisibility.inline.value))
    run_imp = RunImportance(agent_meta.get("run_importance", RunImportance.ephemeral.value))
    inputs = {
        "message": text,
        "attachments": [resource.to_dict() for resource in (attachments or [])],
        "session_id": session_id,
        "user_meta": user_meta or {},
    }

    record = await container.run_manager.submit_run(
        graph_id=graph_id,
        inputs=inputs,
        session_id=session_id,
        identity=identity,
        origin=origin,
        visibility=run_vis,
        importance=run_imp,
        agent_id=agent_id,
        app_id=agent_meta.get("app_id"),
        tags=tags or [f"agent:{agent_id}"],
    )
    return record.run_id
