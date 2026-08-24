"""Shared provider-edge helpers for canonical attachment ingress."""

from __future__ import annotations

from typing import Literal

from aethergraph.contracts.services.channel import OutEvent
from aethergraph.services.integration import ResourceIngressError, ResourceIngressPolicy


def _resource_policy(container) -> ResourceIngressPolicy:
    resource_ingress = getattr(container.integration_ingress, "resource_ingress", None)
    policy = getattr(resource_ingress, "policy", None)
    if not isinstance(policy, ResourceIngressPolicy):
        raise RuntimeError("Canonical integration attachment policy is unavailable.")
    return policy


def _provider_delivery_adapter(container, prefix: str):
    adapter = container.channels.adapters.get(prefix)
    downstream = getattr(adapter, "downstream", None)
    return downstream or adapter


def _attachment_read_budget(
    policy: ResourceIngressPolicy,
    *,
    current_total: int,
    declared_size: int | None,
    attachment_id: str,
) -> tuple[
    int,
    Literal[
        "integration.attachment_too_large",
        "integration.attachment_total_exceeded",
    ],
]:
    if declared_size is not None and declared_size > policy.max_file_bytes:
        raise ResourceIngressError(
            code="integration.attachment_too_large",
            message=f"Attachment {attachment_id!r} exceeds the file limit.",
        )
    if declared_size is not None and current_total + declared_size > policy.max_total_bytes:
        raise ResourceIngressError(
            code="integration.attachment_total_exceeded",
            message="Ingress attachment total exceeds the configured limit.",
        )
    remaining = max(0, policy.max_total_bytes - current_total)
    read_limit = min(policy.max_file_bytes, remaining)
    overflow_code = (
        "integration.attachment_total_exceeded"
        if read_limit < policy.max_file_bytes
        else "integration.attachment_too_large"
    )
    return read_limit, overflow_code


async def _notify_provider_rejection(
    container,
    *,
    prefix: str,
    channel_key: str,
    code: str,
    message: str,
) -> None:
    adapter = _provider_delivery_adapter(container, prefix)
    if adapter is None:
        raise RuntimeError(f"Provider delivery adapter is unavailable for {prefix!r}.")
    await adapter.send(
        OutEvent(
            type="agent.message",
            channel=channel_key,
            text=f"I couldn't accept that message: {message} ({code})",
        )
    )


async def _notify_rejected_receipt(
    container,
    *,
    prefix: str,
    channel_key: str,
    receipt,
) -> None:
    if getattr(receipt, "accepted", False):
        return
    code = str(getattr(receipt, "rejection_code", None) or "integration.ingress_rejected")
    message = str(
        getattr(receipt, "rejection_message", None) or "The message could not be accepted."
    )
    await _notify_provider_rejection(
        container,
        prefix=prefix,
        channel_key=channel_key,
        code=code,
        message=message,
    )
