"""Exact lifecycle ownership for one selected canonical storage bundle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .contracts import (
    StorageBundle,
    StorageCapability,
    StorageConfigurationError,
    StorageConflictError,
    StorageFormatError,
    StorageHealth,
    StorageHealthError,
    StorageOpenRequest,
)
from .provider_registry import StorageProviderRegistry


@dataclass(slots=True)
class StorageComposition:
    """Own exact selection, validation, readiness, and closure for one bundle."""

    registry: StorageProviderRegistry
    required_capabilities: frozenset[StorageCapability] = frozenset()
    _bundle: StorageBundle | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if any(
            not isinstance(capability, StorageCapability)
            for capability in self.required_capabilities
        ):
            raise TypeError("required_capabilities must contain StorageCapability members")
        self.required_capabilities = frozenset(self.required_capabilities)

    async def open(self, request: StorageOpenRequest) -> StorageBundle:
        """Open and validate exactly one selected provider bundle.

        The selected provider constructs its bundle synchronously. Composition then
        verifies bundle identity, access mode, format, required capabilities, and
        asynchronous readiness before publishing the bundle to callers.

        Examples:
            Open a configured local provider:
                ```python
                composition = StorageComposition(registry)
                bundle = await composition.open(request)
                ```

            Require transactional compare-and-set behavior:
                ```python
                composition = StorageComposition(
                    registry,
                    frozenset({StorageCapability.ATOMIC_COMPARE_AND_SET}),
                )
                bundle = await composition.open(request)
                ```

        Args:
            request: Complete trusted request selecting one exact provider.

        Returns:
            StorageBundle: Ready provider-owned bundle retained by this lifecycle owner.

        Notes:
            Any failed selection, construction, or validation permanently closes the
            owner; a partial bundle is closed. No provider retry or fallback occurs.
        """
        async with self._lock:
            if self._closed:
                raise StorageHealthError("Storage composition is already closed")
            if self._bundle is not None:
                raise StorageConflictError("Storage composition already owns a bundle")

            bundle: StorageBundle | None = None
            try:
                provider = self.registry.create(request.selection.provider)
                provider.validate_config(request.selection)
                bundle = provider.open(request)
                self._validate_bundle(request, bundle)
                health = await bundle.health()
                if not health.ready:
                    detail = f": {health.detail}" if health.detail else ""
                    raise StorageHealthError(
                        f"Storage provider {bundle.provider_name!r} is not ready{detail}"
                    )
            except BaseException:
                self._closed = True
                if bundle is not None:
                    await bundle.close()
                raise

            self._bundle = bundle
            return bundle

    async def health(self) -> StorageHealth:
        """Return readiness from the currently owned bundle.

        The lifecycle owner delegates to the exact bundle opened by `open()` and
        never constructs, resolves, or probes another provider.

        Examples:
            Check an active composition:
                ```python
                status = await composition.health()
                ```

            Reject a check before open:
                ```python
                with pytest.raises(StorageHealthError):
                    await StorageComposition(registry).health()
                ```

        Args:
            None.

        Returns:
            StorageHealth: Current readiness of the one owned provider bundle.

        Notes:
            Calling before successful open or after close raises `StorageHealthError`.
        """
        async with self._lock:
            if self._closed or self._bundle is None:
                raise StorageHealthError("Storage composition has no active bundle")
            return await self._bundle.health()

    async def close(self) -> None:
        """Close the owned bundle successfully at most once.

        Closure is serialized with open and health operations. Calling close before
        open is valid and permanently closes this lifecycle owner. A bundle-close
        failure leaves the owner active so the exact same close can be retried.

        Examples:
            Close after runtime shutdown:
                ```python
                await composition.close()
                ```

            Close repeatedly during error cleanup:
                ```python
                await composition.close()
                await composition.close()
                ```

        Args:
            None.

        Returns:
            None: The bundle is closed or the owner was already closed.

        Notes:
            Services must not close individual bundle stores. Failed durable flushes
            are never converted into a closed composition state.
        """
        async with self._lock:
            if self._closed:
                return
            if self._bundle is not None:
                await self._bundle.close()
            self._closed = True

    def _validate_bundle(self, request: StorageOpenRequest, bundle: StorageBundle) -> None:
        if bundle.provider_name != request.selection.provider:
            raise StorageConfigurationError(
                "Selected storage provider returned a bundle with a different provider name"
            )
        if bundle.mode is not request.mode:
            raise StorageConfigurationError(
                "Selected storage provider returned a bundle with a different open mode"
            )
        if bundle.format_version != request.expected_format_version:
            raise StorageFormatError(
                "Selected storage provider returned format "
                f"{bundle.format_version}, expected {request.expected_format_version}"
            )
        bundle.capabilities.require(bundle.provider_name, self.required_capabilities)
