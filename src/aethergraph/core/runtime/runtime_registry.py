from contextvars import ContextVar

from aethergraph.services.registry.unified_registry import UnifiedRegistry

# Single process-wide registry instance
__singleton_registry: UnifiedRegistry = UnifiedRegistry()

# Optional overrides per-context (rarely needed)
_current_registry: ContextVar[UnifiedRegistry | None] = ContextVar("ag_registry", default=None)


def global_registry() -> UnifiedRegistry:
    """
    Return the process-wide global registry instance.

    Use this when you explicitly want the singleton, e.g. wiring into containers.
    """
    return __singleton_registry


def set_current_registry(reg: UnifiedRegistry) -> None:
    """
    Override the registry for the current context (e.g., tests or special scopes).
    """
    _current_registry.set(reg)


def current_registry() -> UnifiedRegistry:
    """
    Get the current registry.

    Priority:
      1) Container services (if installed) and they hold a registry.
      2) ContextVar override (set_current_registry).
      3) Global singleton.
    """
    from .runtime_services import current_services  # lazy import to avoid cycles

    # 1) If services are installed and have a registry, prefer that
    try:
        svc = current_services()
        reg = getattr(svc, "registry", None)
        if isinstance(reg, UnifiedRegistry):
            return reg
    except Exception:
        # services not installed or not accessible in this context
        pass

    # 2) ContextVar
    reg = _current_registry.get()
    if reg is not None:
        return reg

    # 3) Fallback to singleton
    return __singleton_registry
