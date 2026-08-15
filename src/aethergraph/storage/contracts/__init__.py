"""Canonical contracts for selecting and opening AG storage providers."""

from .capabilities import StorageCapabilities, StorageCapability
from .errors import (
    DuplicateStorageProviderError,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageConflictError,
    StorageError,
    StorageFormatError,
    StorageHealthError,
    StorageProviderRegistrationError,
    StorageReadOnlyError,
    StorageScopeError,
    UnknownStorageProviderError,
)
from .provider import (
    StorageBundle,
    StorageClock,
    StorageHealth,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProvider,
    StorageProviderSelection,
    StorageSecretResolver,
)
from .scope import StorageScope

__all__ = [
    "DuplicateStorageProviderError",
    "StorageBundle",
    "StorageCapabilities",
    "StorageCapability",
    "StorageCapabilityError",
    "StorageClock",
    "StorageConfigurationError",
    "StorageConflictError",
    "StorageError",
    "StorageFormatError",
    "StorageHealth",
    "StorageHealthError",
    "StorageOpenMode",
    "StorageOpenRequest",
    "StorageProvider",
    "StorageProviderRegistrationError",
    "StorageProviderSelection",
    "StorageReadOnlyError",
    "StorageScope",
    "StorageScopeError",
    "StorageSecretResolver",
    "UnknownStorageProviderError",
]
