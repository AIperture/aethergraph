"""Exact lifecycle ownership for one selected canonical storage bundle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum, auto
import re
from threading import RLock
from uuid import uuid4

from .contracts import (
    StorageBundle,
    StorageCapability,
    StorageConfigurationError,
    StorageConflictError,
    StorageFormatError,
    StorageHealth,
    StorageHealthError,
    StorageOpenRequest,
    StorageStartupDiagnostic,
    StorageStartupError,
)
from .provider_registry import StorageProviderRegistry


class _CompositionState(Enum):
    NEW = auto()
    PREPARED = auto()
    READY = auto()
    STARTUP_FAILED = auto()
    CLOSED = auto()


@dataclass(slots=True)
class StorageComposition:
    """Own exact selection, validation, readiness, and closure for one bundle."""

    registry: StorageProviderRegistry
    required_capabilities: frozenset[StorageCapability] = frozenset()
    _bundle: StorageBundle | None = field(default=None, init=False, repr=False)
    _request: StorageOpenRequest | None = field(default=None, init=False, repr=False)
    _state: _CompositionState = field(
        default=_CompositionState.NEW,
        init=False,
        repr=False,
    )
    _startup_error: BaseException | None = field(default=None, init=False, repr=False)
    _startup_diagnostic: StorageStartupDiagnostic | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _state_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if any(
            not isinstance(capability, StorageCapability)
            for capability in self.required_capabilities
        ):
            raise TypeError("required_capabilities must contain StorageCapability members")
        self.required_capabilities = frozenset(self.required_capabilities)

    @property
    def startup_diagnostic(self) -> StorageStartupDiagnostic | None:
        """Return the immutable primary startup diagnostic, when startup failed.

        Intro:
            The diagnostic remains available after cleanup and across follow-on calls
            so callers do not lose the provider, data-root, stage, or primary failure.

        Examples:
            Inspect a healthy composition:
                ```python
                assert composition.startup_diagnostic is None
                ```

            Inspect a failed composition:
                ```python
                diagnostic = composition.startup_diagnostic
                assert diagnostic is not None and diagnostic.diagnostic_id
                ```

        Args:
            None.

        Returns:
            StorageStartupDiagnostic | None: Stable immutable failure evidence.

        Notes:
            Cleanup failure augments the same diagnostic without replacing its
            primary stage, exception type, or message.
        """

        with self._state_lock:
            return self._startup_diagnostic

    def prepare(self, request: StorageOpenRequest) -> StorageBundle:
        """Construct exactly one selected bundle without publishing it as ready.

        Intro:
            Resolves one explicitly registered provider, validates its configuration,
            and performs its synchronous construction. Bundle identity, format,
            capability, and asynchronous health validation remain for `start()`.

        Examples:
            Prepare before an application lifespan starts:
                ```python
                composition = StorageComposition(registry)
                bundle = composition.prepare(request)
                ```

            Compose services before readiness:
                ```python
                bundle = composition.prepare(request)
                services = compose_services(bundle)
                ```

        Args:
            request: Complete trusted request selecting one exact provider.

        Returns:
            StorageBundle: The sole constructed bundle retained by this owner.

        Notes:
            The returned bundle is internal composition input, not operationally ready.
            Selection or construction failure is terminal and never selects a fallback.
        """
        with self._state_lock:
            if self._state is _CompositionState.CLOSED:
                if self._startup_diagnostic is not None:
                    raise StorageStartupError(self._startup_diagnostic) from (
                        self._startup_error
                    )
                raise StorageHealthError("Storage composition is already closed")
            if self._state is _CompositionState.STARTUP_FAILED:
                assert self._startup_diagnostic is not None
                raise StorageStartupError(self._startup_diagnostic) from self._startup_error
            if self._state is not _CompositionState.NEW:
                raise StorageConflictError("Storage composition already owns a bundle")

            try:
                provider = self.registry.create(request.selection.provider)
                provider.validate_config(request.selection)
                bundle = provider.open(request)
            except BaseException:
                self._state = _CompositionState.CLOSED
                raise

            self._bundle = bundle
            self._request = request
            self._state = _CompositionState.PREPARED
            return bundle

    async def start(self) -> StorageBundle:
        """Validate readiness and publish the one prepared bundle.

        Intro:
            Performs identity, mode, format, capability, and asynchronous health
            validation at an async runtime boundary. Successful repeated calls return
            the same ready bundle without another health check or provider selection.

        Examples:
            Start during application lifespan:
                ```python
                composition.prepare(request)
                bundle = await composition.start()
                ```

            Reuse idempotent readiness:
                ```python
                first = await composition.start()
                second = await composition.start()
                assert first is second
                ```

        Args:
            None.

        Returns:
            StorageBundle: The exact validated and ready bundle.

        Notes:
            Startup failure is terminal for selection and reopen. Failed cleanup keeps
            the same bundle owned so `close()` can retry without fallback or reselection.
        """
        async with self._lock:
            with self._state_lock:
                if self._state is _CompositionState.READY:
                    assert self._bundle is not None
                    return self._bundle
                if self._state is _CompositionState.CLOSED:
                    if self._startup_diagnostic is not None:
                        raise StorageStartupError(self._startup_diagnostic) from (
                            self._startup_error
                        )
                    raise StorageHealthError("Storage composition is already closed")
                if self._state is _CompositionState.STARTUP_FAILED:
                    assert self._startup_diagnostic is not None
                    raise StorageStartupError(
                        self._startup_diagnostic
                    ) from self._startup_error
                if self._state is _CompositionState.NEW:
                    raise StorageHealthError("Storage composition is not prepared")
                assert self._bundle is not None
                assert self._request is not None
                bundle = self._bundle
                request = self._request

            stage = "bundle_validation"
            try:
                self._validate_bundle(request, bundle)
                stage = "health_check"
                health = await bundle.health()
                if not health.ready:
                    detail = f": {health.detail}" if health.detail else ""
                    raise StorageHealthError(
                        f"Storage provider {bundle.provider_name!r} is not ready{detail}"
                    )
            except BaseException as startup_error:
                diagnostic = StorageStartupDiagnostic(
                    diagnostic_id=f"storage_{uuid4().hex}",
                    workspace_root=request.workspace_root,
                    provider_name=request.selection.provider,
                    stage=stage,
                    exception_type=type(startup_error).__name__,
                    message=_safe_error_message(startup_error),
                )
                with self._state_lock:
                    self._state = _CompositionState.STARTUP_FAILED
                    self._startup_error = startup_error
                    self._startup_diagnostic = diagnostic
                try:
                    await bundle.close()
                except BaseException as cleanup_error:
                    diagnostic = replace(
                        diagnostic,
                        cleanup_exception_type=type(cleanup_error).__name__,
                        cleanup_message=_safe_error_message(cleanup_error),
                    )
                    with self._state_lock:
                        self._startup_diagnostic = diagnostic
                    raise StorageStartupError(diagnostic) from startup_error
                with self._state_lock:
                    self._bundle = None
                    self._request = None
                    self._state = _CompositionState.CLOSED
                raise StorageStartupError(diagnostic) from startup_error

            with self._state_lock:
                self._state = _CompositionState.READY
            return bundle

    async def open(self, request: StorageOpenRequest) -> StorageBundle:
        """Prepare and start exactly one selected provider bundle.

        Intro:
            Preserves the original all-async convenience for callers already at an
            async boundary by composing `prepare()` and `start()` without alternate
            selection or parallel storage construction.

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
            Any failed selection or startup is terminal for reopen. Cleanup remains
            retryable only against the exact retained bundle when its first close fails.
        """
        self.prepare(request)
        return await self.start()

    async def health(self) -> StorageHealth:
        """Return readiness from the currently owned bundle.

        Intro:
            Delegates only after successful asynchronous startup and never turns a
            prepared or failed bundle into an operational one.

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
            with self._state_lock:
                if self._state is not _CompositionState.READY or self._bundle is None:
                    if self._startup_diagnostic is not None:
                        raise StorageStartupError(self._startup_diagnostic) from (
                            self._startup_error
                        )
                    raise StorageHealthError("Storage composition has no active bundle")
                bundle = self._bundle
            return await bundle.health()

    async def close(self) -> None:
        """Close the owned bundle successfully at most once.

        Intro:
            Closes a prepared, ready, or startup-failed bundle through its provider
            lifecycle. A failed close retains the exact bundle and state for retry.

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
            with self._state_lock:
                if self._state is _CompositionState.CLOSED:
                    return
                if self._state is _CompositionState.NEW:
                    self._state = _CompositionState.CLOSED
                    return
                bundle = self._bundle
            if bundle is not None:
                await bundle.close()
            with self._state_lock:
                self._bundle = None
                self._request = None
                self._state = _CompositionState.CLOSED

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


def _safe_error_message(error: BaseException) -> str:
    message = " ".join(str(error).split()) or "Storage startup failed."
    message = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://)([^@\s/]+)@",
        r"\1***@",
        message,
    )
    message = re.sub(
        r"(?i)\b(password|token|secret|api[_-]?key)\s*[=:]\s*[^\s,;]+",
        r"\1=***",
        message,
    )
    return message[:1_000]
