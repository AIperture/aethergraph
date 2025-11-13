from contextlib import AbstractContextManager 

class RunRegistrationGuard(AbstractContextManager):
    """Context manager to register and unregister a scheduler for a run. Primarily for resume handling.
    On enter, registers the scheduler with the container's scheduler registry.
    On exit, unregisters the scheduler.
    """
    def __init__(self, *, run_id: str, scheduler, container):
        self.run_id = run_id
        self.scheduler = scheduler
        self.container = container
        self._did_reg = False

    def __enter__(self):
        reg = self.container.sched_registry
        existing = reg.get(self.run_id)
        if existing is not None and existing is not self.scheduler:
            # Be explicit to avoid silent clobbering
            raise RuntimeError(f"Scheduler already registered for run_id={self.run_id}")
        reg.register(self.run_id, self.scheduler)
        self._did_reg = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._did_reg:
            try:
                self.container.sched_registry.unregister(self.run_id)
            finally:
                self._did_reg = False
        # Return False to propagate any exception (important so callers can detect failures)
        return False