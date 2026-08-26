"""Typed failures raised by the canonical storage-provider boundary."""

from __future__ import annotations

from .diagnostics import StorageStartupDiagnostic


class StorageError(RuntimeError):
    """Base class for failures owned by the storage-provider boundary."""


class StorageConfigurationError(StorageError, ValueError):
    """Selected provider configuration is invalid or incomplete."""


class StorageProviderRegistrationError(StorageConfigurationError):
    """A provider name or registration conflicts with the exact registry."""


class DuplicateStorageProviderError(StorageProviderRegistrationError):
    """The registry already contains the exact provider name."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(f"Storage provider is already registered: {provider_name!r}")
        self.provider_name = provider_name


class UnknownStorageProviderError(StorageProviderRegistrationError):
    """The selected exact provider name is not registered."""

    def __init__(self, provider_name: str) -> None:
        super().__init__(f"Unknown storage provider: {provider_name!r}")
        self.provider_name = provider_name


class StorageCapabilityError(StorageConfigurationError):
    """A selected provider lacks a capability required at open time."""

    def __init__(self, provider_name: str, missing: tuple[str, ...]) -> None:
        rendered = ", ".join(missing)
        super().__init__(
            f"Storage provider {provider_name!r} lacks required capabilities: {rendered}"
        )
        self.provider_name = provider_name
        self.missing = missing


class StorageScopeError(StorageError, ValueError):
    """A storage operation has malformed or missing canonical scope identity."""


class StorageConflictError(StorageError):
    """An atomic write failed because its expected version was stale."""


class StorageCapacityError(StorageError):
    """A bounded provider admission queue has no capacity for another write."""


class StorageIntegrityError(StorageError):
    """Persisted or staged content failed an integrity requirement."""


class StorageNotFoundError(StorageError):
    """A required exact storage record or blob does not exist."""


class StorageReadOnlyError(StorageError):
    """A write was attempted through a read-only provider bundle."""


class StorageFormatError(StorageError):
    """A workspace manifest or provider schema version is unsupported."""


class StorageHealthError(StorageError):
    """A provider failed its required readiness or health check."""


class StorageStartupError(StorageHealthError):
    """A canonical storage startup failed with immutable diagnostic evidence."""

    def __init__(self, diagnostic: StorageStartupDiagnostic) -> None:
        """Bind one immutable startup diagnostic to a typed storage failure.

        Intro:
            The exception keeps the same diagnostic available across cleanup and
            follow-on lifecycle calls while preserving the primary failure as cause.

        Examples:
            Raise a typed startup failure:
                ```python
                raise StorageStartupError(diagnostic)
                ```

            Read stable evidence at a Host boundary:
                ```python
                assert error.diagnostic.diagnostic_id
                ```

        Args:
            diagnostic: Immutable primary and optional cleanup failure evidence.

        Returns:
            None: The exception instance is initialized.

        Notes:
            Provider configuration and credentials must never enter the diagnostic.
        """

        self.diagnostic = diagnostic
        super().__init__(
            f"Storage startup {diagnostic.diagnostic_id} failed during "
            f"{diagnostic.stage}: {diagnostic.message}"
        )


class StorageTimeoutError(StorageError, TimeoutError):
    """A bounded storage wait did not reach its required cursor or state."""
