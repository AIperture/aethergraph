"""Shared attachment validation and artifact materialization for canonical ingress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aethergraph.contracts.integration import (
    ExternalSessionBinding,
    IngressAttachment,
    IngressEnvelope,
    IntegrationRoute,
)
from aethergraph.services.channel.resources import (
    ArtifactIngressScope,
    InputResource,
    ResourceEnricher,
    ResourceSet,
    ResourceStager,
)
from aethergraph.storage.contracts import StorageNotFoundError, StorageScope

from .context import VerifiedIntegrationContext


class ResourceIngressError(RuntimeError):
    """Structured failure raised when inbound attachments violate route policy."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.attachments_unsupported",
            "integration.attachment_count_exceeded",
            "integration.attachment_too_large",
            "integration.attachment_total_exceeded",
            "integration.attachment_type_rejected",
            "integration.attachment_bytes_missing",
            "integration.attachment_size_mismatch",
            "integration.attachment_duplicate",
            "integration.artifact_not_found",
            "integration.attachment_scope_invalid",
        ],
        message: str,
    ) -> None:
        """Create one stable ResourceIngress failure.

        Attachment validation fails before root dispatch or continuation resume.

        Examples:
            Reject unsupported attachments:
            ```python
            ResourceIngressError(
                code="integration.attachments_unsupported",
                message="The route does not accept attachments.",
            )
            ```

            Reject an oversized attachment:
            ```python
            ResourceIngressError(
                code="integration.attachment_too_large",
                message="Attachment exceeds the configured limit.",
            )
            ```

        Args:
            code: Stable machine-readable resource failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            No attachment is silently dropped, renamed, or downgraded.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResourceIngressPolicy:
    """Bounded attachment limits applied uniformly to every integration."""

    max_count: int = 10
    max_file_bytes: int = 25 * 1024 * 1024
    max_total_bytes: int = 50 * 1024 * 1024
    allowed_content_types: tuple[str, ...] = ()


class ResourceIngress:
    """Validate and materialize provider-neutral attachments exactly once."""

    def __init__(self, *, container, policy: ResourceIngressPolicy | None = None) -> None:
        """Bind attachment materialization to one AG Host container.

        The service owns artifact scope and limits; transport edges only provide
        authenticated identities and protected bytes.

        Examples:
            Create with default limits:
            ```python
            ingress = ResourceIngress(container=container)
            ```

            Create with a stricter local policy:
            ```python
            ingress = ResourceIngress(
                container=container,
                policy=ResourceIngressPolicy(max_count=2),
            )
            ```

        Args:
            container: AG Host container owning artifact storage and indexing.
            policy: Optional bounded attachment policy.

        Returns:
            None.

        Notes:
            Provider bytes are never fetched by this service from unverified URLs.
        """
        self.container = container
        self.policy = policy or ResourceIngressPolicy()

    async def materialize(
        self,
        *,
        verified: VerifiedIntegrationContext,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        session_scope: StorageScope,
        envelope: IngressEnvelope,
    ) -> tuple[InputResource, ...]:
        """Validate and materialize all attachments from one ingress envelope.

        Provider files are staged under the bound AG session. Existing artifact
        references are validated and enriched without copying their content.

        Examples:
            Materialize provider uploads:
            ```python
            resources = await ingress.materialize(
                verified=verified,
                route=route,
                binding=binding,
                envelope=envelope,
            )
            ```

            Process an envelope without attachments:
            ```python
            assert await ingress.materialize(
                verified=verified,
                route=route,
                binding=binding,
                envelope=text_envelope,
            ) == ()
            ```

        Args:
            verified: Authenticated transport context containing protected bytes.
            route: Exact resolved route and capability requirements.
            binding: Durable external-to-AG session binding.
            session_scope: Canonical scope from the persisted bound session.
            envelope: Closed canonical ingress envelope.

        Returns:
            tuple[InputResource, ...]: Materialized or validated artifact resources.

        Notes:
            Every declared provider attachment must have one exact verified byte payload.
        """
        if session_scope.session_id != binding.ag_session_id:
            raise ResourceIngressError(
                code="integration.attachment_scope_invalid",
                message="The bound session and attachment storage scope do not match.",
            )
        attachments = envelope.attachments
        if not attachments:
            if verified.attachments:
                raise ResourceIngressError(
                    code="integration.attachment_bytes_missing",
                    message="Verified attachment bytes were not declared by the envelope.",
                )
            return ()
        if not route.required_capabilities.attachments:
            raise ResourceIngressError(
                code="integration.attachments_unsupported",
                message=f"Route {route.route_id!r} does not accept attachments.",
            )
        self._validate_declared(attachments)
        verified_bytes = self._verified_bytes(verified)
        resources = ResourceSet()
        actual_total = 0
        for attachment in attachments:
            self._validate_attachment(attachment)
            if attachment.source_kind == "provider_file":
                data = verified_bytes.pop(attachment.attachment_id, None)
                if data is None:
                    raise ResourceIngressError(
                        code="integration.attachment_bytes_missing",
                        message=(
                            f"Authenticated bytes are missing for attachment "
                            f"{attachment.attachment_id!r}."
                        ),
                    )
                if attachment.size_bytes is not None and len(data) != attachment.size_bytes:
                    raise ResourceIngressError(
                        code="integration.attachment_size_mismatch",
                        message=f"Attachment {attachment.attachment_id!r} size does not match.",
                    )
                if len(data) > self.policy.max_file_bytes:
                    raise ResourceIngressError(
                        code="integration.attachment_too_large",
                        message=f"Attachment {attachment.attachment_id!r} exceeds the file limit.",
                    )
                actual_total += len(data)
                if actual_total > self.policy.max_total_bytes:
                    raise ResourceIngressError(
                        code="integration.attachment_total_exceeded",
                        message="Ingress attachment total exceeds the configured limit.",
                    )
                try:
                    resource = await ResourceStager(
                        container=self.container,
                        storage_scope=session_scope,
                    ).stage_bytes(
                        data,
                        name=attachment.filename,
                        mime=attachment.content_type,
                        file_id=attachment.source_id,
                        scope=ArtifactIngressScope(
                            source=verified.integration_kind.value,
                            session_id=binding.ag_session_id,
                            channel_key=envelope.origin_address.channel_key,
                            conversation_id=envelope.external_identity.conversation_id,
                            graph_id="integration",
                            node_id="resource_ingress",
                            tool_name="integration.resource_ingress",
                        ),
                        labels={
                            "integration_id": verified.integration_id,
                            "route_id": route.route_id,
                            "attachment_id": attachment.attachment_id,
                        },
                    )
                except StorageNotFoundError as exc:
                    raise ResourceIngressError(
                        code="integration.attachment_scope_invalid",
                        message="Attachment storage rejected the canonical session scope.",
                    ) from exc
                resources.add(resource)
            else:
                resources.add(
                    InputResource(
                        kind="artifact",
                        source=verified.integration_kind.value,
                        status="materialized",
                        id=attachment.attachment_id,
                        name=attachment.filename,
                        mime=attachment.content_type,
                        size=attachment.size_bytes,
                        artifact_id=attachment.source_id,
                    )
                )
        if verified_bytes:
            raise ResourceIngressError(
                code="integration.attachment_bytes_missing",
                message="Verified attachment bytes contain undeclared attachment identities.",
            )
        enriched = await ResourceEnricher(container=self.container).enrich(resources)
        await self._require_artifacts(enriched)
        return tuple(enriched.resources)

    def _validate_declared(self, attachments: tuple[IngressAttachment, ...]) -> None:
        if len(attachments) > self.policy.max_count:
            raise ResourceIngressError(
                code="integration.attachment_count_exceeded",
                message=f"Ingress contains more than {self.policy.max_count} attachments.",
            )
        ids = [attachment.attachment_id for attachment in attachments]
        if len(ids) != len(set(ids)):
            raise ResourceIngressError(
                code="integration.attachment_duplicate",
                message="Ingress attachment identifiers must be unique.",
            )
        declared_total = sum(attachment.size_bytes or 0 for attachment in attachments)
        if declared_total > self.policy.max_total_bytes:
            raise ResourceIngressError(
                code="integration.attachment_total_exceeded",
                message="Ingress attachment total exceeds the configured limit.",
            )

    def _validate_attachment(self, attachment: IngressAttachment) -> None:
        if attachment.size_bytes is not None and attachment.size_bytes > self.policy.max_file_bytes:
            raise ResourceIngressError(
                code="integration.attachment_too_large",
                message=f"Attachment {attachment.attachment_id!r} exceeds the file limit.",
            )
        allowed = self.policy.allowed_content_types
        if allowed and attachment.content_type not in allowed:
            raise ResourceIngressError(
                code="integration.attachment_type_rejected",
                message=f"Content type {attachment.content_type!r} is not allowed.",
            )

    @staticmethod
    def _verified_bytes(verified: VerifiedIntegrationContext) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for item in verified.attachments:
            if item.attachment_id in out:
                raise ResourceIngressError(
                    code="integration.attachment_duplicate",
                    message="Verified attachment identifiers must be unique.",
                )
            out[item.attachment_id] = item.data
        return out

    async def _require_artifacts(self, resources: ResourceSet) -> None:
        get_artifact = getattr(getattr(self.container, "artifact_index", None), "get", None)
        if not callable(get_artifact):
            raise ResourceIngressError(
                code="integration.artifact_not_found",
                message="ResourceIngress requires an artifact index.",
            )
        for resource in resources:
            if not resource.artifact_id or await get_artifact(resource.artifact_id) is None:
                raise ResourceIngressError(
                    code="integration.artifact_not_found",
                    message=f"Artifact {resource.artifact_id!r} does not exist.",
                )
