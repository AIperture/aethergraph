from contextvars import ContextVar

from aethergraph.contracts.services.metering import MeteringService

MeterContext = dict[str, object]
current_meter_context: ContextVar[MeterContext | None] = ContextVar(
    "ag_meter_context",
    default=None,
)

_current_metering: ContextVar[MeteringService | None] = ContextVar("ag_metering", default=None)


def set_current_metering(service: MeteringService | None) -> None:
    """Set the metering service for the current execution context.

    Intro:
        Installs or clears one context-local metering implementation without
        creating a process-wide fallback service.

    Examples:
        Install a service:
        ```python
        set_current_metering(meter)
        ```

        Clear a service:
        ```python
        set_current_metering(None)
        ```

    Args:
        service: Metering implementation for the current context, or `None`.

    Returns:
        None: The context variable is updated before returning.

    Notes:
        Runtime container metering takes precedence in `current_metering`.
    """
    _current_metering.set(service)


def current_metering() -> MeteringService | None:
    """Resolve the canonical metering implementation for the current runtime.

    Intro:
        Prefers the active runtime container's metering service, then checks the
        explicit context-local override. Absence remains explicit as `None`.

    Examples:
        Record usage when configured:
        ```python
        meter = current_metering()
        if meter is not None:
            await meter.record_run(run_id="run-1", status="succeeded")
        ```

        Detect an unconfigured context:
        ```python
        if current_metering() is None:
            print("metering unavailable")
        ```

    Args:
        None.

    Returns:
        MeteringService | None: Active metering service, or `None` when unconfigured.

    Notes:
        This function does not construct a no-op implementation or mutate context.
    """
    try:
        from .runtime_services import current_services

        service = getattr(current_services(), "metering", None)
        if service is not None:
            return service
    except Exception:
        pass
    return _current_metering.get()
