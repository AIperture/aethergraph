from __future__ import annotations

from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.services.channel.attachments import InputAttachment, attachment_to_dict


def attachments_from_incoming_files(
    files: list[dict[str, Any]] | None,
    *,
    source: str,
) -> list[InputAttachment]:
    out: list[InputAttachment] = []
    for file_info in files or []:
        artifact_id = file_info.get("artifact_id") or file_info.get("id")
        out.append(
            InputAttachment(
                kind="artifact",
                source=source,
                artifact_id=artifact_id,
                name=file_info.get("name"),
                mimetype=file_info.get("mimetype"),
                size=file_info.get("size"),
                uri=file_info.get("uri"),
                url=file_info.get("url"),
                meta=file_info.get("extra") or {},
            )
        )
    return out


async def dispatch_channel_turn_run(
    *,
    container: Any,
    identity: RequestIdentity | None,
    agent_id: str,
    text: str,
    attachments: list[InputAttachment] | None,
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
        "attachments": [attachment_to_dict(a) for a in (attachments or [])],
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
