from __future__ import annotations

from typing import Any
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.contracts.integration import OriginBinding
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
    origin_channel_key: str,
    integration_id: str,
    route_id: str,
    external_conversation_id: str,
    external_thread_id: str | None,
    capability_profile_id: str,
    user_meta: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    origin: RunOrigin = RunOrigin.chat,
) -> str:
    """Submit one channel-originated root turn with an immutable reply origin.

    Examples:
        Dispatch an AG UI turn:
        ```python
        run_id = await dispatch_channel_turn_run(
            container=container,
            identity=identity,
            agent_id="assistant",
            text="Hello",
            attachments=[],
            session_id="session-1",
            origin_channel_key="ui:session/session-1",
            integration_id="ag-ui",
            route_id="agent:assistant",
            external_conversation_id="session-1",
            external_thread_id=None,
            capability_profile_id="ag-ui-v1",
        )
        ```

        Dispatch a Slack thread:
        ```python
        run_id = await dispatch_channel_turn_run(
            container=container,
            identity=identity,
            agent_id="assistant",
            text="Hello",
            attachments=[],
            origin_channel_key="slack:team/T:chan/C",
            integration_id="slack",
            route_id="agent:assistant",
            external_conversation_id="slack:C#thread:100.1",
            external_thread_id="100.1",
            capability_profile_id="slack-v1",
        )
        ```

    Args:
        container: Service container owning the registry and run manager.
        identity: Authenticated request identity, or `None` for local mode.
        agent_id: Exact registered agent selected by the transport route.
        text: Normalized inbound message text.
        attachments: Normalized inbound resources.
        session_id: Existing AG session identifier, when already bound.
        origin_channel_key: Exact adapter address for replies.
        integration_id: Integration instance that accepted the message.
        route_id: Exact route that selected the agent.
        external_conversation_id: Provider-neutral external conversation identity.
        external_thread_id: Optional external thread identity.
        capability_profile_id: Validated adapter capability profile.
        user_meta: Normalized metadata supplied to the agent.
        tags: Run tags to persist.
        origin: Root run origin classification.

    Returns:
        Identifier of the submitted root run.

    Notes:
        The supplied address fields are closed into `OriginBinding` before
        submission. The run never consults a process-global Channel default.
    """
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
    effective_session_id = session_id or f"session-{uuid4().hex[:12]}"
    origin_binding = OriginBinding(
        integration_id=integration_id,
        route_id=route_id,
        session_id=effective_session_id,
        channel_key=origin_channel_key,
        external_conversation_id=external_conversation_id,
        external_thread_id=external_thread_id,
        capability_profile_id=capability_profile_id,
    )
    run_vis = RunVisibility(agent_meta.get("run_visibility", RunVisibility.inline.value))
    run_imp = RunImportance(agent_meta.get("run_importance", RunImportance.ephemeral.value))
    inputs = {
        "message": text,
        "attachments": [resource.to_dict() for resource in (attachments or [])],
        "session_id": effective_session_id,
        "user_meta": user_meta or {},
    }

    record = await container.run_manager.submit_run(
        graph_id=graph_id,
        inputs=inputs,
        session_id=effective_session_id,
        identity=identity,
        origin=origin,
        visibility=run_vis,
        importance=run_imp,
        agent_id=agent_id,
        app_id=agent_meta.get("app_id"),
        tags=tags or [f"agent:{agent_id}"],
        run_config={"origin_binding": origin_binding.model_dump(mode="json")},
    )
    return record.run_id
