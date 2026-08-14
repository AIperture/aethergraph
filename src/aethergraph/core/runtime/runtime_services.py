from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Any
import warnings

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.core.runtime.base_service import Service

_current = ContextVar("aeg_services", default=None)
# process-wide fallback (handles contextvar boundary issues)
_services_global: Any = None
# allow registering external services before main services are ready
_pending_ext_services: dict[str, Any] = {}

_REMOVED_FIRST_CLASS_SERVICE_NAMES = frozenset(
    {
        "execute",
        "execution",
        "harness",
        "kb",
        "knowledge",
        "mcp",
        "planner",
        "planning",
        "skills",
        "web_search",
        "websearch",
    }
)


_pending_lock = RLock()

# Ordered operations (some things depend on earlier steps)
_pending_ops_order: list[str] = []
# Keyed storage so repeated registrations overwrite instead of duplicating
_pending_ops: dict[str, Callable[[Any], Any]] = {}
# Optional: store results if you want “handles” later
_pending_results: dict[str, Any] = {}


def _defer_op(key: str, op: Callable[[Any], Any]) -> None:
    """Register (or replace) a deferred operation."""
    with _pending_lock:
        if key not in _pending_ops:
            _pending_ops_order.append(key)
        _pending_ops[key] = op


def _flush_pending_ops(services: Any) -> None:
    """Apply all deferred operations once services exist."""
    with _pending_lock:
        keys = list(_pending_ops_order)
        _pending_ops_order.clear()
        ops = _pending_ops.copy()
        _pending_ops.clear()

    for key in keys:
        op = ops.get(key)
        if op is None:
            continue
        try:
            _pending_results[key] = op(services)
        except Exception:
            # You can choose to log here instead of raising,
            # but raising is usually better so startup fails loudly.
            raise


def _try_apply_or_defer(key: str, fn: Callable[[Any], Any]) -> Any | None:
    """
    If services installed: run now and return result.
    Else: defer it and return None.
    """
    try:
        svc = current_services()
    except RuntimeError:
        _defer_op(key, fn)
        return None
    else:
        return fn(svc)


def install_services(services: Any) -> None:
    global _services_global, _pending_ext_services
    _services_global = services

    # Attach pending ext services (your existing behavior)
    ext = getattr(services, "ext_services", None)
    if isinstance(ext, dict) and _pending_ext_services:
        for name, svc in _pending_ext_services.items():
            ext.setdefault(name, svc)
        _pending_ext_services = {}

    # NEW: apply all other pending mutations
    _flush_pending_ops(services)

    return _current.set(services)


def ensure_services_installed(factory: Callable[[], Any]) -> Any:
    global _services_global, _pending_ext_services
    svc = _current.get() or _services_global
    if svc is None:
        svc = factory()
        _services_global = svc

        # hydrate pending external services
        ext = getattr(svc, "ext_services", None)
        if isinstance(ext, dict) and _pending_ext_services:
            for name, s in _pending_ext_services.items():
                ext.setdefault(name, s)
            _pending_ext_services = {}

        # NEW: apply pending ops on first creation too
        _flush_pending_ops(svc)

    _current.set(svc)
    return svc


def current_services() -> Any:
    svc = _current.get() or _services_global
    if svc is None:
        raise RuntimeError(
            "No services installed. Call install_services(container) at app startup."
        )
    return svc


@contextmanager
def use_services(services):
    tok = _current.set(services)
    try:
        yield
    finally:
        _current.reset(tok)


# --------- Channel service helpers ---------
def get_channel_service() -> Any:
    svc = current_services()
    return svc.channels  # ChannelBus


def register_channel_adapter(name: str, adapter: Any) -> None:
    svc = current_services()
    svc.channels.register_adapter(name, adapter)


# --------- LLM service helpers ---------
def get_llm_service() -> Any:
    svc = current_services()
    return svc.llm


def register_llm_client(
    profile: str,
    provider: str,
    model: str,
    embed_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> LLMClientProtocol | None:
    """Register Chat and optional legacy embedding profiles independently.

    Intro:
        Configures the named Chat client immediately when services are installed,
        or defers the same operation until installation. A supplied legacy
        `embed_model` is routed to the separate embedding service.

    Examples:
        Register one Chat profile:
            ```python
            register_llm_client("default", "openai", "gpt-5-mini")
            ```

        Preserve a legacy combined registration:
            ```python
            register_llm_client(
                "search",
                "openai",
                "gpt-5-mini",
                embed_model="text-embedding-3-small",
            )
            ```

    Args:
        profile: Exact Chat and optional embedding profile name.
        provider: Registered provider identity.
        model: Chat model identity.
        embed_model: Deprecated embedding model compatibility input.
        base_url: Optional provider base URL override.
        api_key: Optional in-memory provider credential.
        timeout: Optional HTTP timeout in seconds.

    Returns:
        LLMClientProtocol | None: Configured Chat client when services are
            installed, otherwise `None` after deferring registration.

    Notes:
        `embed_model` remains only as a public migration boundary. New code must
        configure embeddings through `NodeContext.embedding()` settings or the
        embedding service. Missing embedding services fail before Chat mutation.
    """

    normalized_embed_model = str(embed_model or "").strip() or None
    if normalized_embed_model is not None:
        warnings.warn(
            "register_llm_client(embed_model=...) is deprecated; configure an "
            "independent embedding profile instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _op(svc: Any) -> LLMClientProtocol:
        embed_service = getattr(svc, "embed_service", None)
        if normalized_embed_model is not None and embed_service is None:
            raise RuntimeError(
                "Legacy embed_model registration requires an enabled embedding service."
            )
        client = svc.llm.configure_profile(
            profile=profile,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        if normalized_embed_model is not None:
            embed_service.configure_profile(
                name=profile,
                provider=provider,
                model=normalized_embed_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        return client

    key = f"llm_client:profile={profile}:provider={provider}:model={model}"
    return _try_apply_or_defer(key, _op)


# backend compatibility
set_llm_client = register_llm_client


# --------- Logger helpers ---------
def current_logger_factory() -> Any:
    svc = current_services()
    return svc.logger


# --------- External context services ---------
def register_context_service(name: str, service: Service) -> None:
    """
    Register an external service for NodeContext access.

    This function attaches an external service to the current service container
    under the specified name. If no container is installed yet, the service is
    stashed in a pending registry and will be attached automatically when
    install_services() is called.

    Examples:
        Register a custom database service:
        ```python
        register_context_service("mydb", MyDatabaseService())
        ```

    Args:
        name: The unique string identifier for the external service.
        service: The service instance to register.

    Returns:
        None

    Notes:
        - If called before install_services(), the service will be attached later.
        - Services are accessible via NodeContext.ext_services[name].
    """
    global _pending_ext_services

    if name in _REMOVED_FIRST_CLASS_SERVICE_NAMES:
        raise ValueError(
            f"External context service name {name!r} is reserved for a removed "
            "first-class capability; expose it through an Engine Tool plugin instead."
        )

    try:
        svc = current_services()
    except RuntimeError:
        # No container yet: keep it in the staging area.
        _pending_ext_services[name] = service
        return

    # Container exists: attach immediately.
    svc.ext_services[name] = service


def get_ext_context_service(name: str) -> Service:
    """
    Retrieve an external context service by name.

    This function returns the external service registered under the given name
    from the current service container's ext_services registry.

    Examples:
        Access a registered service:
        ```python
        mydb = get_ext_context_service("mydb")
        ```

    Args:
        name: The string name of the external service to retrieve.

    Returns:
        The service instance registered under the given name, or None if not found.

    Raises:
        RuntimeError: If no services container is installed.
    """
    if name in _REMOVED_FIRST_CLASS_SERVICE_NAMES:
        raise KeyError(f"Removed first-class capability is not available: {name}")
    svc = current_services()
    return svc.ext_services.get(name)


def list_ext_context_services() -> list[str]:
    """
    List all registered external context service names.

    This function returns a list of all names for services currently registered
    in the ext_services registry of the current service container.

    Examples:
        List all available external services:
        ```python
        services = list_ext_context_services()
        print(services)
        ```

    Args:
        None

    Returns:
        A list of strings representing the names of all registered external services.
        Returns an empty list if no services are registered.

    Raises:
        RuntimeError: If no services container is installed.
    """
    svc = current_services()
    return list(svc.ext_services.keys())
