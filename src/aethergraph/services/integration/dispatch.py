"""Canonical AG root-turn dispatch behind unified integration ingress."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import logging
from typing import Protocol

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.contracts.integration import (
    AgentInputResource,
    ExternalSessionBinding,
    IngressEnvelope,
    IntegrationRoute,
    OriginBinding,
)
from aethergraph.core.runtime.run_types import (
    RunAdmissionError,
    RunImportance,
    RunOrigin,
    RunVisibility,
)
from aethergraph.services.channel.resources import InputResource
from aethergraph.storage.contracts import StorageScope

from .context import VerifiedIntegrationContext
from .delivery import SemanticTurnMonitor

_LOG = logging.getLogger("aethergraph.integration.dispatch")


class RootTurnDispatcher(Protocol):
    """Provider-neutral contract for starting one bound AG root turn."""

    async def start(
        self,
        *,
        verified: VerifiedIntegrationContext,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        session_scope: StorageScope,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
        admission_callback: Callable[[str], Awaitable[None]] | None = None,
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
            session_scope: Exact canonical scope of the provisioned AG session.
            envelope: Closed canonical ingress envelope.
            resources: Materialized inbound resources.
            admission_callback: Optional Host callback that must persist the accepted
                turn identity before execution can start.

        Returns:
            str: Submitted AG root run identifier used as the turn identity.

        Notes:
            Implementations must propagate `OriginBinding` through `run_config`.
        """
        ...


class AGRootTurnDispatcher:
    """Submit canonical integration turns through the AG RunManager."""

    def __init__(self, container, *, turn_monitor: SemanticTurnMonitor) -> None:
        """Bind root dispatch to one AG Host container.

        Agent selection remains entirely in the resolved `IntegrationRoute`.

        Examples:
            Create a dispatcher:
            ```python
            dispatcher = AGRootTurnDispatcher(container, turn_monitor=monitor)
            ```

            Install it in the coordinator:
            ```python
            coordinator = IntegrationIngressCoordinator(
                root_dispatcher=AGRootTurnDispatcher(
                    container,
                    turn_monitor=monitor,
                ),
                **dependencies,
            )
            ```

        Args:
            container: AG Host container owning registry and RunManager services.
            turn_monitor: Canonical observer for terminal semantic state.

        Returns:
            None.

        Notes:
            The dispatcher does not read provider configuration or default agents.
        """
        self.container = container
        self.turn_monitor = turn_monitor

    async def start(
        self,
        *,
        verified: VerifiedIntegrationContext,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        session_scope: StorageScope,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
        admission_callback: Callable[[str], Awaitable[None]] | None = None,
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
            session_scope: Exact canonical scope of the provisioned AG session.
            envelope: Closed canonical ingress envelope.
            resources: Materialized inbound resources.
            admission_callback: Optional Host callback that must persist the accepted
                turn identity before execution can start.

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
        agent_input_resources: list[AgentInputResource] = []
        for resource in resources:
            attachment_id = resource.labels.get("attachment_id")
            if not isinstance(attachment_id, str) or not attachment_id.strip():
                raise ValueError(
                    "Materialized ingress resource has no canonical attachment identity."
                )
            agent_input_resources.append(
                AgentInputResource(
                    attachment_id=attachment_id,
                    artifact_id=str(resource.artifact_id),
                    filename=(resource.name or resource.id or str(resource.artifact_id)),
                    content_type=(resource.mime or "application/octet-stream"),
                    size_bytes=resource.size,
                )
            )

        async def admit(record) -> None:
            expected_dimensions = {
                "session_id": binding.ag_session_id,
                "graph_id": graph_id,
                "agent_id": route.entry_agent_id,
            }
            actual_dimensions = {
                "session_id": record.session_id,
                "graph_id": record.graph_id,
                "agent_id": record.agent_id,
            }
            mismatched_dimensions = {
                name: {"expected": expected, "actual": actual_dimensions[name]}
                for name, expected in expected_dimensions.items()
                if actual_dimensions[name] != expected
            }
            if mismatched_dimensions:
                error = RunAdmissionError(
                    run_id=record.run_id,
                    code="integration.artifact_run_scope_mismatch",
                    stage="run_attachment_admission",
                    safe_message=(
                        "The persisted AG run scope does not match the integration route."
                    ),
                    details={"mismatched_dimensions": mismatched_dimensions},
                )
                _LOG.error(
                    "integration.artifact_admission_failed",
                    extra={
                        "integration_error_code": error.code,
                        "error_stage": error.stage,
                        "run_id": record.run_id,
                        "integration_id": envelope.integration_id,
                        "route_id": route.route_id,
                        **error.details,
                    },
                )
                raise error
            run_dimensions = {
                "session_id": record.session_id,
                "run_id": record.run_id,
                "graph_id": record.graph_id,
                "agent_id": record.agent_id,
                "org_id": record.org_id,
                "user_id": record.user_id,
            }
            scope_values = session_scope.as_filter()
            owner_mismatches = {
                name: {"expected": expected, "actual": run_dimensions[name]}
                for name, expected in scope_values.items()
                if name in run_dimensions
                and run_dimensions[name] is not None
                and run_dimensions[name] != expected
            }
            if owner_mismatches:
                error = RunAdmissionError(
                    run_id=record.run_id,
                    code="integration.artifact_run_scope_mismatch",
                    stage="run_attachment_admission",
                    safe_message=(
                        "The persisted AG run ownership does not match its integration session."
                    ),
                    details={"mismatched_dimensions": owner_mismatches},
                )
                _LOG.error(
                    "integration.artifact_admission_failed",
                    extra={
                        "integration_error_code": error.code,
                        "error_stage": error.stage,
                        "run_id": record.run_id,
                        "integration_id": envelope.integration_id,
                        "route_id": route.route_id,
                        **error.details,
                    },
                )
                raise error
            scope_values.update(
                {name: value for name, value in run_dimensions.items() if value is not None}
            )
            run_scope = StorageScope(**scope_values)
            artifacts = self.container.artifact_factory.for_execution(
                run_scope,
                tool_name="integration.resource_admission",
            )
            try:
                for resource, agent_input_resource in zip(
                    resources, agent_input_resources, strict=True
                ):
                    if not resource.artifact_id:
                        raise RunAdmissionError(
                            run_id=record.run_id,
                            code="integration.artifact_identity_missing",
                            stage="run_attachment_admission",
                            safe_message="An admitted input resource has no artifact identity.",
                            details={"attachment_id": resource.id or resource.name or "unknown"},
                        )
                    attachment_identity = str(agent_input_resource.attachment_id)
                    digest = hashlib.sha256(
                        "\x00".join(
                            (record.run_id, attachment_identity, resource.artifact_id)
                        ).encode("utf-8")
                    ).hexdigest()
                    await artifacts.attach_existing(
                        resource.artifact_id,
                        occurrence_id=f"occurrence-input-{digest}",
                        occurred_at=record.started_at,
                        labels={
                            "attachment_id": attachment_identity,
                            "integration_id": envelope.integration_id,
                            "route_id": route.route_id,
                        },
                    )
            except RunAdmissionError:
                raise
            except Exception as exc:
                error = RunAdmissionError(
                    run_id=record.run_id,
                    code="integration.artifact_run_attachment_rejected",
                    stage="run_attachment_admission",
                    safe_message=("Canonical artifact storage rejected a run attachment."),
                    details={
                        "storage_error_type": type(exc).__name__,
                        "run_scope": run_scope.as_filter(),
                        "artifact_ids": tuple(resource.artifact_id for resource in resources),
                    },
                )
                _LOG.exception(
                    "integration.artifact_admission_failed",
                    extra={
                        "integration_error_code": error.code,
                        "error_stage": error.stage,
                        "run_id": record.run_id,
                        "integration_id": envelope.integration_id,
                        "route_id": route.route_id,
                        **error.details,
                    },
                )
                raise error from exc
            if admission_callback is not None:
                await admission_callback(record.run_id)

        record = await self.container.run_manager.submit_run(
            graph_id=graph_id,
            inputs={
                "input": envelope.input.model_copy(
                    update={"resources": tuple(agent_input_resources)}
                ).model_dump(mode="json"),
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
            admission_callback=(admit if resources or admission_callback is not None else None),
        )
        self.turn_monitor.observe(
            run_id=record.run_id,
            session_id=binding.ag_session_id,
            route_id=route.route_id,
            integration_id=envelope.integration_id,
        )
        return record.run_id
