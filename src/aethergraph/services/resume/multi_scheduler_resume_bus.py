import asyncio
from datetime import UTC, datetime
from logging import getLogger

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.resume import ResumeBus
from aethergraph.services.continuations.continuation import Continuation, ContinuationStatus
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

    async def enqueue_resume(self, *, continuation: Continuation, payload: dict) -> None:
        """Deliver one validated continuation to its active local scheduler.

        Intro:
            Validates the durable continuation, dispatches exactly once on the
            scheduler's owning loop, and deletes the continuation after success.

        Examples:
            Deliver from the scheduler's event loop:
            ```python
            await bus.enqueue_resume(continuation=wait, payload={"ok": True})
            ```

            Handle a run that is not active in this process:
            ```python
            try:
                await bus.enqueue_resume(continuation=wait, payload={})
            except SchedulerUnavailableError:
                schedule_recovery("run-2")
            ```

        Args:
            continuation: Exact tokenless continuation authorized by the caller.
            payload: Resume data delivered to the waiting node.

        Returns:
            None: Dispatch and post-success terminal persistence are complete.

        Notes:
            Scheduler absence, dispatch failure, and terminal-write failure are
            explicit errors. Before successful dispatch, the durable
            continuation remains available for retry or recovery.
        """
        current = await self.store.get_by_id(
            continuation.run_id,
            continuation.node_id,
            continuation.continuation_id,
        )
        if current is None or current.revision != continuation.revision or current.closed:
            raise PermissionError("Continuation is no longer waiting")
        continuation = current
        run_id = continuation.run_id
        node_id = continuation.node_id

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
                await self.store.close(
                    continuation,
                    status=ContinuationStatus.RESUMED,
                    closed_at=datetime.now(UTC),
                )
            except Exception:
                self.logger.exception(
                    "[multi-resume-bus] failed to close continuation for %s/%s",
                    run_id,
                    node_id,
                )
                raise
