"""Exact capability admission for a complete AetherGraph runtime bundle."""

from __future__ import annotations

from .composition import StorageComposition
from .contracts import StorageCapability
from .provider_registry import StorageProviderRegistry

RUNTIME_STORAGE_CAPABILITIES = frozenset(
    {
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
        StorageCapability.ATOMIC_COMPARE_AND_SET,
        StorageCapability.ORDERED_APPEND,
        StorageCapability.MONOTONIC_CURSORS,
        StorageCapability.TTL,
        StorageCapability.LEASES,
        StorageCapability.BLOB_STREAMING,
        StorageCapability.BLOB_RANGE_READ,
        StorageCapability.SEARCH_STRUCTURAL,
        StorageCapability.SEARCH_LEXICAL,
        StorageCapability.HEALTH,
    }
)


def create_runtime_storage_composition(
    registry: StorageProviderRegistry,
) -> StorageComposition:
    """Create one lifecycle owner with complete runtime capability admission.

    Intro:
        Fixes the provider behaviors required by the full AG runtime before selection,
        preparation, service binding, or publication. Optional semantic and hybrid
        search remain explicit provider capabilities and are never fallback targets.

    Examples:
        Prepare a built-in local runtime:
            ```python
            composition = create_runtime_storage_composition(local_registry)
            bundle = composition.prepare(open_request)
            ```

        Reject an incomplete external provider during startup:
            ```python
            composition = create_runtime_storage_composition(external_registry)
            composition.prepare(open_request)
            await composition.start()
            ```

    Args:
        registry: Exact explicit provider registry owned by runtime composition.

    Returns:
        StorageComposition: New lifecycle owner requiring the full runtime capability set.

    Notes:
        The function performs no provider lookup, open, health check, fallback, or I/O.
        Capability validation occurs in `StorageComposition.start()` before the bundle
        becomes operationally ready or is published to service consumers.
    """
    return StorageComposition(
        registry=registry,
        required_capabilities=RUNTIME_STORAGE_CAPABILITIES,
    )
