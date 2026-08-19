"""Single canonical ingress coordination boundary for every integration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Literal
from uuid import uuid4

from aethergraph.contracts.integration import (
    ArtifactAvailablePayload,
    ExternalIdentity,
    HostManifest,
    IngressEnvelope,
    IngressReceipt,
    InputAcceptedPayload,
    SemanticEventKind,
)

from .context import VerifiedIntegrationContext
from .delivery import SemanticEventEmitter
from .dispatch import RootTurnDispatcher
from .event_contracts import InboundEventStore
from .idempotency import IngressIdempotencyStore
from .interactions import (
    InteractionResolutionError,
    InteractionResolver,
    build_interaction_payload,
)
from .resources import ResourceIngress, ResourceIngressError
from .routes import IntegrationRouteError, ManifestRouteResolver
from .session_bindings import (
    IntegrationSessionResolution,
    IntegrationSessionStore,
    SessionBindingError,
)


class IngressCoordinatorError(RuntimeError):
    """Structured operational failure raised outside terminal receipt handling."""

    def __init__(
        self,
        *,
        code: Literal["integration.ingress_in_progress"],
        message: str,
    ) -> None:
        """Create one stable coordinator operational failure.

        Operational failures do not start or resume an AG turn.

        Examples:
            Report a concurrent duplicate:
            ```python
            IngressCoordinatorError(
                code="integration.ingress_in_progress",
                message="The original ingress is still running.",
            )
            ```

            Return the stable code to a transport:
            ```python
            try:
                receipt = await coordinator.accept(verified=verified, envelope=envelope)
            except IngressCoordinatorError as exc:
                return {"code": exc.code}
            ```

        Args:
            code: Stable machine-readable coordinator failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            The transport may retry the same idempotency identity later.
        """
        super().__init__(message)
        self.code = code


class IntegrationIngressCoordinator:
    """Accept canonical ingress and choose exactly one resume or root dispatch."""

    def __init__(
        self,
        *,
        manifest: HostManifest,
        route_resolver: ManifestRouteResolver,
        idempotency_store: IngressIdempotencyStore,
        session_store: IntegrationSessionStore,
        resource_ingress: ResourceIngress,
        interaction_resolver: InteractionResolver,
        inbound_events: InboundEventStore,
        semantic_emitter: SemanticEventEmitter,
        resume_router,
        root_dispatcher: RootTurnDispatcher,
    ) -> None:
        """Bind the one ingress transaction boundary for an AG Host deployment.

        Intro:
            Every transport invokes this service object directly after authentication
            and canonical envelope construction.

        Examples:
            Create a local Host coordinator:
            ```python
            coordinator = IntegrationIngressCoordinator(
                manifest=manifest,
                route_resolver=ManifestRouteResolver(manifest),
                idempotency_store=idempotency_store,
                session_store=session_store,
                resource_ingress=resource_ingress,
                interaction_resolver=interaction_resolver,
                inbound_events=inbound_events,
                semantic_emitter=semantic_emitter,
                resume_router=container.resume_router,
                root_dispatcher=root_dispatcher,
            )
            ```

            Share one coordinator across HTTP and provider runners:
            ```python
            app.state.integration_ingress = coordinator
            slack_runner.integration_ingress = coordinator
            ```

        Args:
            manifest: Immutable Host manifest and deployment authority.
            route_resolver: Exact immutable-manifest route resolver.
            idempotency_store: Durable ingress claim and receipt store.
            session_store: Atomic canonical integration-session provisioner.
            resource_ingress: Shared attachment validation/materialization service.
            interaction_resolver: Exact open-interaction resolver.
            inbound_events: Focused durable ingress writer.
            semantic_emitter: Canonical endpoint/provider semantic event emitter.
            resume_router: AG continuation resume router.
            root_dispatcher: Exact route-selected AG root dispatcher.

        Returns:
            None.

        Notes:
            No dependency is optional and no secondary execution path is consulted.
        """
        self.manifest = manifest
        self.route_resolver = route_resolver
        self.idempotency_store = idempotency_store
        self.session_store = session_store
        self.resource_ingress = resource_ingress
        self.interaction_resolver = interaction_resolver
        self.inbound_events = inbound_events
        self.semantic_emitter = semantic_emitter
        self.semantic_events = semantic_emitter.store
        self.resume_router = resume_router
        self.root_dispatcher = root_dispatcher

    async def provision_session(
        self,
        *,
        route_id: str,
        external_identity: ExternalIdentity,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
        title: str | None = None,
    ) -> IntegrationSessionResolution:
        """Provision one canonical session through the manifest-owned route.

        Host adapters that need a session before submitting ingress use this same
        boundary as normal coordinator acceptance. The route, build, scope, session,
        and external binding therefore have one source of truth.

        Examples:
            Provision a Studio thread:
                ```python
                result = await coordinator.provision_session(
                    route_id="studio-ai",
                    external_identity=identity,
                    binding_id="binding-1",
                    ag_session_id="session-1",
                    now=now,
                    title="New chat",
                )
                ```

            Repair an existing binding whose session was never created:
                ```python
                repaired = await coordinator.provision_session(
                    route_id="studio-ai",
                    external_identity=identity,
                    binding_id="unused-binding",
                    ag_session_id="unused-session",
                    now=later,
                )
                ```

        Args:
            route_id: Exact enabled route in the immutable Host manifest.
            external_identity: Authenticated external conversation identity.
            binding_id: Candidate binding identifier used only on first creation.
            ag_session_id: Candidate session identifier used only on first creation.
            now: Authoritative UTC provisioning timestamp.
            title: Optional title used only when creating the canonical session.

        Returns:
            IntegrationSessionResolution: Canonical binding and creation ownership.

        Notes:
            This is the only public pre-ingress session creation boundary.
        """
        route = self.route_resolver.require(route_id)
        return await self.session_store.provision(
            route=route,
            external_identity=external_identity,
            build_id=self.manifest.build_id,
            binding_id=binding_id,
            ag_session_id=ag_session_id,
            now=now,
            title=title,
        )

    async def accept(
        self,
        *,
        verified: VerifiedIntegrationContext,
        envelope: IngressEnvelope,
        root_admission_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> IngressReceipt:
        """Accept one ingress envelope and perform exactly one terminal action.

        The coordinator deduplicates, resolves route and session, materializes
        resources, persists ingress, then resumes one interaction or starts one
        root turn. Stable pre-dispatch failures are persisted as rejected receipts.

        Examples:
            Accept AG UI text:
            ```python
            receipt = await coordinator.accept(
                verified=verified_ui,
                envelope=envelope,
            )
            ```

            Accept a provider callback:
            ```python
            receipt = await coordinator.accept(
                verified=verified_slack,
                envelope=choice_envelope,
            )
            ```

        Args:
            verified: Authenticated integration authority from the transport edge.
            envelope: Closed canonical ingress envelope.
            root_admission_callback: Optional Host callback that persists a new root
                turn binding before RunManager schedules its execution.

        Returns:
            IngressReceipt: Durable original or duplicate terminal result.

        Notes:
            A concurrent duplicate raises `integration.ingress_in_progress` and
            performs no side effect. Post-dispatch persistence failures remain
            pending for explicit Host reconciliation rather than retrying a path.
        """
        claim = await self.idempotency_store.claim(
            deployment_id=self.manifest.deployment_id,
            envelope=envelope,
        )
        if claim.receipt is not None:
            return claim.receipt
        if not claim.acquired:
            raise IngressCoordinatorError(
                code="integration.ingress_in_progress",
                message="The original ingress operation is still in progress.",
            )

        route = None
        binding = None
        try:
            route = self.route_resolver.resolve(verified=verified, envelope=envelope)
            binding_resolution = await self.session_store.provision(
                route=route,
                external_identity=envelope.external_identity,
                build_id=self.manifest.build_id,
                binding_id=f"binding-{uuid4().hex}",
                ag_session_id=f"session-{uuid4().hex}",
                now=envelope.received_at,
            )
            binding = binding_resolution.binding
            resources = await self.resource_ingress.materialize(
                verified=verified,
                route=route,
                binding=binding,
                session_scope=binding_resolution.session.scope,
                envelope=envelope,
            )
            resolved = await self.interaction_resolver.resolve(
                binding=binding,
                envelope=envelope,
            )
        except (
            IntegrationRouteError,
            SessionBindingError,
            ResourceIngressError,
            InteractionResolutionError,
        ) as exc:
            receipt = IngressReceipt(
                accepted=False,
                duplicate=False,
                action="rejected",
                deployment_id=self.manifest.deployment_id,
                route_id=route.route_id if route is not None else None,
                session_id=binding.ag_session_id if binding is not None else None,
                rejection_code=exc.code,
            )
            await self.idempotency_store.complete(
                deployment_id=self.manifest.deployment_id,
                envelope=envelope,
                receipt=receipt,
            )
            return receipt

        inbound = await self.inbound_events.append(
            deployment_id=self.manifest.deployment_id,
            route=route,
            binding=binding,
            envelope=envelope,
            resources=resources,
        )
        await self.semantic_emitter.emit_semantic(
            session_id=binding.ag_session_id,
            turn_id=inbound.event_id,
            producer=f"integration.{route.integration_kind.value}",
            kind=SemanticEventKind.INPUT_ACCEPTED,
            payload=InputAcceptedPayload(
                input_id=envelope.idempotency_key,
                text=envelope.text,
                artifacts=tuple(
                    ArtifactAvailablePayload(
                        artifact_id=resource.artifact_id,
                        filename=resource.name or resource.artifact_id,
                        content_type=resource.mime or "application/octet-stream",
                        size_bytes=resource.size or 0,
                    )
                    for resource in resources
                    if resource.artifact_id is not None
                ),
                interaction_id=envelope.choice.interaction_id if envelope.choice else None,
                option_ids=envelope.choice.option_ids if envelope.choice else (),
            ),
            extensions={"aethergraph.route_id": route.route_id},
            timestamp=envelope.received_at,
        )
        try:
            if resolved is not None:
                continuation = resolved.continuation
                await self.resume_router.resume_continuation(
                    continuation,
                    build_interaction_payload(
                        resolved=resolved,
                        envelope=envelope,
                        resources=resources,
                    ),
                )
                action = "continuation_resumed"
                turn_id = continuation.run_id
            else:
                turn_id = await self.root_dispatcher.start(
                    verified=verified,
                    route=route,
                    binding=binding,
                    envelope=envelope,
                    resources=resources,
                    admission_callback=root_admission_callback,
                )
                action = "root_turn_started"
        except Exception:
            receipt = IngressReceipt(
                accepted=False,
                duplicate=False,
                action="rejected",
                deployment_id=self.manifest.deployment_id,
                route_id=route.route_id,
                session_id=binding.ag_session_id,
                rejection_code="integration.dispatch_failed",
                event_cursor=inbound.cursor,
            )
            await self.idempotency_store.complete(
                deployment_id=self.manifest.deployment_id,
                envelope=envelope,
                receipt=receipt,
            )
            raise

        receipt = IngressReceipt(
            accepted=True,
            duplicate=False,
            action=action,
            deployment_id=self.manifest.deployment_id,
            route_id=route.route_id,
            session_id=binding.ag_session_id,
            turn_id=turn_id,
            event_cursor=inbound.cursor,
        )
        await self.idempotency_store.complete(
            deployment_id=self.manifest.deployment_id,
            envelope=envelope,
            receipt=receipt,
        )
        return receipt
