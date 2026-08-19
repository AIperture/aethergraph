"""Exact integration-route resolution for one immutable host manifest."""

from __future__ import annotations

from typing import Literal

from aethergraph.contracts.integration import (
    HostManifest,
    IngressEnvelope,
    IntegrationRoute,
)

from .context import VerifiedIntegrationContext


class IntegrationRouteError(RuntimeError):
    """Structured failure raised when canonical ingress cannot resolve one route."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.identity_mismatch",
            "integration.route_not_found",
            "integration.route_disabled",
            "integration.route_ambiguous",
        ],
        message: str,
    ) -> None:
        """Create one stable route-resolution failure.

        Examples:
            Report an unauthenticated identity mismatch:
            ```python
            IntegrationRouteError(
                code="integration.identity_mismatch",
                message="Verified integration does not match the envelope.",
            )
            ```

            Report ambiguous manifest configuration:
            ```python
            IntegrationRouteError(
                code="integration.route_ambiguous",
                message="More than one route accepted the envelope.",
            )
            ```

        Args:
            code: Stable machine-readable route failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Route failures never select another integration or entry agent.
        """
        super().__init__(message)
        self.code = code


class ManifestRouteResolver:
    """Resolve exactly one enabled route from an immutable host manifest."""

    def __init__(self, manifest: HostManifest) -> None:
        """Index immutable integration routes for exact lookup.

        Examples:
            Build a resolver from a validated manifest:
            ```python
            resolver = ManifestRouteResolver(manifest)
            ```

            Reuse one resolver for every ingress accepted by a host:
            ```python
            route = resolver.resolve(verified=verified, envelope=envelope)
            ```

        Args:
            manifest: Closed host manifest containing the only route authority.

        Returns:
            None.

        Notes:
            The resolver does not mutate routes or maintain a second registry.
        """
        self.manifest = manifest

    def resolve(
        self,
        *,
        verified: VerifiedIntegrationContext,
        envelope: IngressEnvelope,
    ) -> IntegrationRoute:
        """Resolve one route after checking authenticated integration identity.

        Examples:
            Resolve an endpoint route:
            ```python
            route = resolver.resolve(verified=verified, envelope=envelope)
            ```

            Handle a stable route error:
            ```python
            try:
                route = resolver.resolve(verified=verified, envelope=envelope)
            except IntegrationRouteError as exc:
                rejection_code = exc.code
            ```

        Args:
            verified: Authenticated integration and tenant identity from the edge.
            envelope: Closed canonical ingress envelope.

        Returns:
            IntegrationRoute: The one enabled route that accepts the envelope.

        Notes:
            Empty match-policy fields are wildcards. Non-empty fields are exact
            allowlists. Zero and multiple matches are both failures.
        """
        if (
            verified.integration_id != envelope.integration_id
            or verified.external_tenant_id != envelope.external_identity.tenant_id
        ):
            raise IntegrationRouteError(
                code="integration.identity_mismatch",
                message="Verified integration identity does not match the ingress envelope.",
            )

        integration_routes = tuple(
            route
            for route in self.manifest.integration_routes
            if route.integration_id == verified.integration_id
            and route.integration_kind is verified.integration_kind
        )
        candidates = integration_routes

        if envelope.route_hint is not None:
            candidates = tuple(
                route for route in candidates if route.route_id == envelope.route_hint
            )
        if envelope.endpoint_id is not None:
            candidates = tuple(
                route for route in candidates if route.endpoint_id == envelope.endpoint_id
            )

        matched = tuple(route for route in candidates if self._matches(route, envelope))
        enabled = tuple(route for route in matched if route.enabled)
        if len(enabled) == 1:
            return enabled[0]
        if len(enabled) > 1:
            raise IntegrationRouteError(
                code="integration.route_ambiguous",
                message="Ingress matched more than one enabled integration route.",
            )
        if matched:
            raise IntegrationRouteError(
                code="integration.route_disabled",
                message="The exact integration route is disabled.",
            )
        raise IntegrationRouteError(
            code="integration.route_not_found",
            message="No integration route accepts the authenticated ingress identity.",
        )

    def require(self, route_id: str) -> IntegrationRoute:
        """Return one enabled route by its exact manifest identity.

        Direct Host APIs use this lookup when authentication and route matching were
        already completed by their own closed boundary.

        Examples:
            Resolve a Studio route:
                ```python
                route = resolver.require("studio-ai")
                ```

            Handle a disabled route:
                ```python
                try:
                    route = resolver.require("disabled-route")
                except IntegrationRouteError as exc:
                    assert exc.code == "integration.route_disabled"
                ```

        Args:
            route_id: Exact route identifier from the immutable Host manifest.

        Returns:
            IntegrationRoute: The one enabled route with the requested identity.

        Notes:
            This lookup does not perform identity matching or choose another route.
        """
        route = next(
            (item for item in self.manifest.integration_routes if item.route_id == route_id),
            None,
        )
        if route is None:
            raise IntegrationRouteError(
                code="integration.route_not_found",
                message=f"Integration route {route_id!r} does not exist.",
            )
        if not route.enabled:
            raise IntegrationRouteError(
                code="integration.route_disabled",
                message=f"Integration route {route_id!r} is disabled.",
            )
        return route

    @staticmethod
    def _matches(route: IntegrationRoute, envelope: IngressEnvelope) -> bool:
        policy = route.match_policy
        identity = envelope.external_identity
        return (
            (not policy.external_tenant_ids or identity.tenant_id in policy.external_tenant_ids)
            and (
                not policy.external_conversation_ids
                or identity.conversation_id in policy.external_conversation_ids
            )
            and (not policy.external_user_ids or identity.user_id in policy.external_user_ids)
        )
