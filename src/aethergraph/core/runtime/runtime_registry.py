from contextvars import ContextVar
from logging import warning
from aethergraph.services.registry.unified_registry import UnifiedRegistry  

__singleton_registry: UnifiedRegistry = UnifiedRegistry()
_current_registry: ContextVar[UnifiedRegistry | None] = ContextVar("ag_registry", default=None)

def set_current_registry(reg: UnifiedRegistry):
    """Set the current registry in contextvar."""
    _current_registry.set(reg)

def current_registry() -> UnifiedRegistry:
    """Get the current registry from contextvar, or raise if not set."""
    # first try if services has a registry set
    from .runtime_services import current_services
    svc = None
    try:
        # get current services and registry from there
        svc = current_services()
        if hasattr(svc, 'registry') and svc.registry is not None:
            return svc.registry
    except Exception as e:
        pass 

    # otherwise use contextvar
    reg = _current_registry.get()
    if reg is None:
        return __singleton_registry # fallback to singleton if not set in local context
    return reg