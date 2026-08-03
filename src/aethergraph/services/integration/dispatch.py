"""Canonical AG root-turn dispatch behind unified integration ingress."""

from __future__ import annotations

from typing import Protocol

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.contracts.integration import (
    ExternalSessionBinding,
    IngressEnvelope,
    IntegrationRoute,
    OriginBinding,
)
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.services.channel.resources import InputResource

from .context import VerifiedIntegrationContext


class RootTurnDispatcher(Protocol):
    """Provider-neutral contract for starting one bound AG root turn."""

    async def start(
        self,
        *,
        verified: VerifiedIntegrationContext,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
    ) -> str:
        """Start one root turn selected only by an immutable manifest route.

        Examples:
            Start a text turn:
            ```python
            turn_id = await dispatcher.start(
                verified=verified,
                route=route,
                binding=binding,
                envelope=envelope,
                resources=(),
            )
            ```

            Start a turn with attachments:
            ```python
            turn_id = await dispatcher.start(
                verified=verified,
                route=route,
                binding=binding,
                envelope=envelope,
                resources=resources,
            )
            ```

        Args:
            verified: Authenticated integration context.
            route: Exact immutable manifest route selecting the entry agent.
            binding: Durable external-to-AG session binding.
            envelope: Closed canonical ingress envelope.
            resources: Materialized inbound resources.

        Returns:
            str: Submitted AG root run identifier used as the turn identity.

        Notes:
            Implementations must propagate `OriginBinding` through `run_config`.
        """
        ...


class AGRootTurnDispatcher:
    """Submit canonical integration turns through the AG RunManager."""

    def __init__(self, container) -> None:
        """Bind root dispatch to one AG Host container.

        Agent selection remains entirely in the resolved `IntegrationRoute`.

        Examples:
            Create a dispatcher:
            ```python
            dispatcher = AGRootTurnDispatcher(container)
            ```

            Install it in the coordinator:
            ```python
            coordinator = IntegrationIngressCoordinator(
                root_dispatcher=AGRootTurnDispatcher(container),
                **dependencies,
            )
            ```

        Args:
            container: AG Host container owning registry and RunManager services.

        Returns:
            None.

        Notes:
            The dispatcher does not read provider configuration or default agents.
        """
        self.container = container

    async def start(
        self,
        *,
        verified: VerifiedIntegrationContext,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
    ) -> str:
        """Submit one exact route-selected root turn with immutable origin.

        The run inputs contain provider-neutral text, attachments, session, and
        bounded transport metadata. The external provider never selects a graph.

        Examples:
            Start an AG UI turn:
            ```python
            turn_id = await dispatcher.start(
                verified=verified_ui,
                route=ui_route,
                binding=binding,
                envelope=envelope,
                resources=(),
            )
            ```

            Start a Slack attachment turn:
            ```python
            turn_id = await dispatcher.start(
                verified=verified_slack,
                route=slack_route,
                binding=binding,
                envelope=envelope,
                resources=resources,
            )
            ```

        Args:
            verified: Authenticated integration context.
            route: Exact manifest route selecting the entry agent.
            binding: Durable external-to-AG session binding.
            envelope: Closed canonical ingress envelope.
            resources: Materialized inbound resources.

        Returns:
            str: Submitted AG run identifier.

        Notes:
            Only `graphfn`-backed agents are supported by this initial Host path.
        """
        identity = verified.request_identity or RequestIdentity(
            user_id="local",
            org_id="local",
            mode="local",
        )
        registry = scoped_registry(identity)
        agent_meta = registry.get_meta(nspace="agent", name=route.entry_agent_id)
        if not agent_meta:
            raise ValueError(f"Agent not found: {route.entry_agent_id}")
        backing = agent_meta.get("backing", {})
        if backing.get("type") != "graphfn":
            raise ValueError(
                f"Unsupported agent backing type: {backing.get('type')}. "
                "Only 'graphfn' is supported."
            )
        graph_id = backing["name"]
        origin_binding = OriginBinding(
            integration_id=envelope.integration_id,
            route_id=route.route_id,
            session_id=binding.ag_session_id,
            channel_key=envelope.origin_address.channel_key,
            external_conversation_id=envelope.external_identity.conversation_id,
            external_thread_id=envelope.external_identity.thread_id,
            capability_profile_id=envelope.origin_address.capability_profile_id,
        )
        user_meta = dict(envelope.transport_metadata)
        if envelope.structured_input is not None:
            user_meta["structured_input"] = envelope.structured_input
        record = await self.container.run_manager.submit_run(
            graph_id=graph_id,
            inputs={
                "message": envelope.text or "",
                "attachments": [resource.to_dict() for resource in resources],
                "session_id": binding.ag_session_id,
                "user_meta": user_meta,
            },
            session_id=binding.ag_session_id,
            identity=identity,
            origin=RunOrigin.chat,
            visibility=RunVisibility(agent_meta.get("run_visibility", RunVisibility.inline.value)),
            importance=RunImportance(
                agent_meta.get("run_importance", RunImportance.ephemeral.value)
            ),
            agent_id=route.entry_agent_id,
            app_id=agent_meta.get("app_id"),
            tags=[f"agent:{route.entry_agent_id}", f"route:{route.route_id}"],
            run_config={"origin_binding": origin_binding.model_dump(mode="json")},
        )
        return record.run_id
