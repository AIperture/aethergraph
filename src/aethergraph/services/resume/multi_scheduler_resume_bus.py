import asyncio
import hmac
from logging import getLogger

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.resume import ResumeBus
from aethergraph.services.schedulers.registry import SchedulerRegistry

log = getLogger(__name__)


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
        cont = await self.store.get(run_id, node_id)
        if not cont or not hmac.compare_digest(cont.token, token):
            self.logger.warning(
                "[multi-resume-bus] invalid continuation/token for %s/%s", run_id, node_id
            )
            return

        sched = self.registry.get(run_id)
        if not sched:
            self.logger.warning("[multi-resume-bus] no scheduler for run_id=%s", run_id)
            return

        loop = getattr(sched, "loop", None)
        if loop is None:
            self.logger.error(
                "[multi-resume-bus] scheduler.loop is not set yet for run_id=%s", run_id
            )
            return

        # Always post to the scheduler's loop
        fut = asyncio.run_coroutine_threadsafe(
            sched.on_resume_event(run_id, node_id, payload), loop
        )
        try:
            await asyncio.wrap_future(fut)
        except Exception as e:
            self.logger.error("[multi-resume-bus] dispatch failed: %s", e, exc_info=True)
            return

        if self.delete_after_resume:
            try:
                await self.store.delete(run_id, node_id)
            except Exception as e:
                self.logger.warning(
                    "[multi-resume-bus] failed to delete continuation for %s/%s: %s",
                    run_id,
                    node_id,
                    e,
                )

        sched.post_resume_event_threadsafe(run_id, node_id, payload)
        return
