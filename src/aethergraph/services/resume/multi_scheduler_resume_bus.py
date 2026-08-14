import asyncio
import hmac
from logging import getLogger

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.resume import ResumeBus
from aethergraph.services.schedulers.registry import SchedulerRegistry

log = getLogger(__name__)


class SchedulerUnavailableError(RuntimeError):
    """Raised when a durable continuation has no active in-process scheduler."""


class MultiSchedulerResumeBus(ResumeBus):
    def __init__(
        self,
        *,
        registry: SchedulerRegistry,
        store: AsyncContinuationStore,
        delete_after_resume: bool = True,
        logger=None,
    ):
        self.registry = registry
        self.store = store
        self.delete_after_resume = delete_after_resume
        self.logger = logger or log

    async def enqueue_resume(self, *, run_id: str, node_id: str, token: str, payload: dict) -> None:
        """Deliver one validated continuation to its active local scheduler.

        Intro:
            Validates the durable continuation, dispatches exactly once on the
            scheduler's owning loop, and deletes the continuation after success.

        Examples:
            Deliver from the scheduler's event loop:
            ```python
            await bus.enqueue_resume(
                run_id="run-1", node_id="approval", token=token, payload={"ok": True}
            )
            ```

            Handle a run that is not active in this process:
            ```python
            try:
                await bus.enqueue_resume(
                    run_id="run-2", node_id="input", token=token, payload={}
                )
            except SchedulerUnavailableError:
                schedule_recovery("run-2")
            ```

        Args:
            run_id: Exact durable run identity.
            node_id: Exact waiting node identity.
            token: Secret continuation token to validate.
            payload: Resume data delivered to the waiting node.

        Returns:
            None: Dispatch and post-success continuation deletion are complete.

        Notes:
            Validation, scheduler absence, dispatch failure, and deletion failure
            are explicit errors. Before successful dispatch, the durable
            continuation remains available for retry or recovery.
        """
        cont = await self.store.get(run_id, node_id)
        if not cont or not hmac.compare_digest(cont.token, token):
            self.logger.warning(
                "[multi-resume-bus] invalid continuation/token for %s/%s", run_id, node_id
            )
            raise PermissionError("Invalid continuation or token")

        sched = self.registry.get(run_id)
        if not sched:
            self.logger.error("[multi-resume-bus] no active scheduler for run_id=%s", run_id)
            raise SchedulerUnavailableError(
                f"No active scheduler for durable continuation {run_id}/{node_id}"
            )

        loop = sched.loop
        if loop is None:
            self.logger.error(
                "[multi-resume-bus] scheduler.loop is not set yet for run_id=%s", run_id
            )
            raise SchedulerUnavailableError(f"Scheduler loop is unavailable for run {run_id}")

        try:
            if loop is asyncio.get_running_loop():
                await sched.on_resume_event(run_id, node_id, payload)
            else:
                future = asyncio.run_coroutine_threadsafe(
                    sched.on_resume_event(run_id, node_id, payload),
                    loop,
                )
                await asyncio.wrap_future(future)
        except Exception:
            self.logger.exception(
                "[multi-resume-bus] dispatch failed for %s/%s",
                run_id,
                node_id,
            )
            raise

        if self.delete_after_resume:
            try:
                await self.store.delete(run_id, node_id)
            except Exception:
                self.logger.exception(
                    "[multi-resume-bus] failed to delete continuation for %s/%s",
                    run_id,
                    node_id,
                )
                raise
