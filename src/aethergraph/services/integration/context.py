"""Verified transport context accepted by unified integration ingress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.integration import IntegrationKind


@dataclass(frozen=True, slots=True)
class VerifiedAttachment:
    """Authenticated provider attachment bytes paired with a declared attachment ID."""

    attachment_id: str
    data: bytes


@dataclass(frozen=True, slots=True)
class VerifiedIntegrationContext:
    """Authenticated integration authority supplied by one transport edge."""

    integration_id: str
    integration_kind: IntegrationKind
    external_tenant_id: str
    attachments: tuple[VerifiedAttachment, ...] = ()
    request_identity: Any | None = None
