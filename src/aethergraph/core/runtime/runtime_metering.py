from contextvars import ContextVar

from aethergraph.contracts.services.metering import MeteringService
from aethergraph.services.metering.noop import NoopMeteringService

MeterContext = dict[str, str | None]
current_meter_context: ContextVar[MeterContext] = ContextVar("ag_meter_context", default={})


# Process-wide default (can be replaced during app startup)
__singleton_metering: MeteringService = NoopMeteringService()

# Optional per-context override
_current_metering: ContextVar[MeteringService | None] = ContextVar("ag_metering", default=None)


def install_global_metering(svc: MeteringService) -> None:
    """
    Called at server startup to install the real metering service.

    E.g. in create_app():
      install_global_metering(EventLogMeteringService(meter_store))
    """
    global __singleton_metering
    __singleton_metering = svc


def set_current_metering(svc: MeteringService) -> None:
    """
    Override the metering service for the current context (tests, special scopes).
    """
    _current_metering.set(svc)


def global_metering() -> MeteringService:
    """
    Return the process-wide singleton (usually a real service after startup,
    or NoopMeteringService in CLI/tests).
    """
    return __singleton_metering


def current_metering() -> MeteringService:
    """
    Get the current metering service.

    Priority:
      1) Container services (if installed) and they hold a .metering
      2) ContextVar override (set_current_metering)
      3) Global singleton (Noop by default)
    """
    from ..runtime.runtime_services import current_services  # lazy import

    # 1) Prefer container services.metering if present
    try:
        svc_container = current_services()
        svc = getattr(svc_container, "metering", None)
        # install the metering from services container
        set_current_metering(svc)
        if isinstance(svc, MeteringService.__constraints__):  # type: ignore[attr-defined]
            return svc
    except Exception:
        pass

    # 2) ContextVar
    svc = _current_metering.get()
    if svc is not None:
        return svc

    # 3) Fallback
    return __singleton_metering
