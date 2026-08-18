"""Exact explicit registry for storage-provider factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from threading import RLock

from .contracts import (
    DuplicateStorageProviderError,
    StorageConfigurationError,
    StorageProvider,
    UnknownStorageProviderError,
)

StorageProviderFactory = Callable[[], StorageProvider]
_PROVIDER_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


class StorageProviderRegistry:
    """Map exact trusted provider names to factories without discovery or fallback."""

    def __init__(
        self,
        providers: Mapping[str, StorageProviderFactory] | None = None,
    ) -> None:
        self._providers: dict[str, StorageProviderFactory] = {}
        self._lock = RLock()
        for name, factory in (providers or {}).items():
            self.register(name, factory)

    def register(self, name: str, factory: StorageProviderFactory) -> None:
        """Register one factory under one exact provider name.

        Registration validates a stable lowercase name and rejects replacement. The
        factory is not called until the selected provider is created.

        Examples:
            Register the local provider:
                ```python
                registry.register("local.sqlite", LocalProvider)
                ```

            Register a company provider:
                ```python
                registry.register("company.postgres-s3", build_company_provider)
                ```

        Args:
            name: Exact trusted provider identifier.
            factory: Zero-argument callable returning a `StorageProvider`.

        Returns:
            None: The factory is registered exactly once.

        Notes:
            Duplicate names raise `DuplicateStorageProviderError`; names are never
            normalized or imported dynamically.
        """
        if not isinstance(name, str) or _PROVIDER_NAME.fullmatch(name) is None:
            raise StorageConfigurationError(
                "storage provider names must be lowercase dot/dash/underscore identifiers"
            )
        if not callable(factory):
            raise StorageConfigurationError("storage provider factory must be callable")
        with self._lock:
            if name in self._providers:
                raise DuplicateStorageProviderError(name)
            self._providers[name] = factory

    def resolve(self, name: str) -> StorageProviderFactory:
        """Resolve the exact registered factory without selecting a default.

        Resolution performs a dictionary lookup only. A missing external provider
        never resolves to the built-in local provider.

        Examples:
            Resolve the local factory:
                ```python
                factory = registry.resolve("local.sqlite")
                ```

            Reject a missing provider:
                ```python
                with pytest.raises(UnknownStorageProviderError):
                    registry.resolve("missing")
                ```

        Args:
            name: Exact selected provider identifier.

        Returns:
            StorageProviderFactory: Registered zero-argument provider factory.

        Notes:
            Unknown names raise `UnknownStorageProviderError`; there is no fallback.
        """
        with self._lock:
            try:
                return self._providers[name]
            except KeyError as exc:
                raise UnknownStorageProviderError(name) from exc

    def create(self, name: str) -> StorageProvider:
        """Create the exact selected provider from its registered factory.

        The new provider's declared name must match the registry key. Configuration
        validation and resource opening remain separate explicit steps.

        Examples:
            Create a selected provider:
                ```python
                provider = registry.create("local.sqlite")
                ```

            Create two independent instances:
                ```python
                first = registry.create("memory")
                second = registry.create("memory")
                ```

        Args:
            name: Exact selected provider identifier.

        Returns:
            StorageProvider: Newly created provider instance.

        Notes:
            A factory returning a mismatched provider name raises
            `StorageConfigurationError`.
        """
        provider = self.resolve(name)()
        if provider.name != name:
            raise StorageConfigurationError(
                f"Storage provider factory {name!r} returned provider {provider.name!r}"
            )
        return provider

    def names(self) -> tuple[str, ...]:
        """Return registered provider names in stable lexical order.

        The returned tuple is detached from the registry and cannot mutate it.

        Examples:
            Inspect an empty registry:
                ```python
                StorageProviderRegistry().names()
                ```

            Inspect registered providers:
                ```python
                assert registry.names() == ("local.sqlite", "memory")
                ```

        Args:
            None.

        Returns:
            tuple[str, ...]: Stable sorted exact provider names.

        Notes:
            Listing names does not instantiate providers.
        """
        with self._lock:
            return tuple(sorted(self._providers))
