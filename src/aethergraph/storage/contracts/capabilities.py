"""Explicit capabilities declared by one selected storage provider."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .errors import StorageCapabilityError


class StorageCapability(StrEnum):
    """Capability names that runtime services may require during provider open."""

    DURABLE = "durable"
    TRANSACTIONS = "transactions"
    ATOMIC_COMPARE_AND_SET = "atomic_compare_and_set"
    ORDERED_APPEND = "ordered_append"
    MONOTONIC_CURSORS = "monotonic_cursors"
    TTL = "ttl"
    LEASES = "leases"
    BLOB_STREAMING = "blob_streaming"
    BLOB_RANGE_READ = "blob_range_read"
    SEARCH_STRUCTURAL = "search_structural"
    SEARCH_SEMANTIC = "search_semantic"
    SEARCH_LEXICAL = "search_lexical"
    SEARCH_HYBRID = "search_hybrid"
    READ_ONLY_OPEN = "read_only_open"
    MIGRATIONS = "migrations"
    HEALTH = "health"


@dataclass(frozen=True, slots=True)
class StorageCapabilities:
    """Immutable set of provider behaviors validated before services are composed."""

    supported: frozenset[StorageCapability]

    @classmethod
    def of(cls, *capabilities: StorageCapability) -> StorageCapabilities:
        """Construct capabilities from exact enum members.

        Duplicate members are collapsed into one immutable set. Strings are not
        coerced, which keeps misspelled capabilities from becoming configuration.

        Examples:
            Declare durable transactions:
                ```python
                StorageCapabilities.of(
                    StorageCapability.DURABLE,
                    StorageCapability.TRANSACTIONS,
                )
                ```

            Declare an ephemeral provider:
                ```python
                StorageCapabilities.of()
                ```

        Args:
            capabilities: Exact `StorageCapability` enum members to declare.

        Returns:
            StorageCapabilities: An immutable capability collection.

        Notes:
            Capability declarations report behavior; they do not enable fallbacks.
        """
        if any(not isinstance(item, StorageCapability) for item in capabilities):
            raise TypeError("capabilities must be StorageCapability members")
        return cls(supported=frozenset(capabilities))

    def supports(self, capability: StorageCapability) -> bool:
        """Report whether this provider declares one exact capability.

        The check has no side effects and does not probe a store implementation.

        Examples:
            Check declared durability:
                ```python
                capabilities.supports(StorageCapability.DURABLE)
                ```

            Check optional hybrid search:
                ```python
                capabilities.supports(StorageCapability.SEARCH_HYBRID)
                ```

        Args:
            capability: Exact capability to inspect.

        Returns:
            bool: `True` only when the capability was declared at open time.

        Notes:
            Services must not use this method to select a fallback backend.
        """
        return capability in self.supported

    def require(
        self,
        provider_name: str,
        required: frozenset[StorageCapability],
    ) -> None:
        """Fail provider open when required capabilities are absent.

        Validation compares immutable declarations only and reports all missing
        capabilities in stable lexical order.

        Examples:
            Require atomic state updates:
                ```python
                capabilities.require(
                    "local.sqlite",
                    frozenset({StorageCapability.ATOMIC_COMPARE_AND_SET}),
                )
                ```

            Accept an empty requirement:
                ```python
                capabilities.require("memory", frozenset())
                ```

        Args:
            provider_name: Exact selected provider name used in an error.
            required: Capabilities required by runtime composition.

        Returns:
            None: Validation succeeds without changing the capability set.

        Notes:
            Missing capabilities raise `StorageCapabilityError`; no alternate provider
            is selected.
        """
        missing = tuple(sorted(item.value for item in required - self.supported))
        if missing:
            raise StorageCapabilityError(provider_name, missing)
