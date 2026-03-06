from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_registry_tenant(
    tenant: Mapping[str, str | None] | None,
) -> dict[str, str | None] | None:
    """
    Canonical registry tenant shape.

    Registry scoping is keyed by org_id + user_id only.
    client_id remains compatibility/observability input and is intentionally excluded.
    """
    if tenant is None:
        return None
    norm = {
        "org_id": tenant.get("org_id"),
        "user_id": tenant.get("user_id"),
    }
    if not any(norm.values()):
        return None
    return norm


def registry_tenant_from_identity(identity: Any) -> dict[str, str | None] | None:
    if identity is None:
        return None
    return normalize_registry_tenant(
        {
            "org_id": getattr(identity, "org_id", None),
            "user_id": getattr(identity, "user_id", None),
        }
    )
