"""Single authorization boundary for continuation resume."""

from __future__ import annotations

from datetime import UTC, datetime
from logging import getLogger
from typing import Any

from jsonschema import ValidationError, validate

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.resume import ResumeBus
from aethergraph.services.continuations.continuation import Continuation, ContinuationStatus

log = getLogger(__name__)


class ResumeRouter:
    """Authorize external tokens once and deliver trusted continuation records."""

    def __init__(
        self, *, store: AsyncContinuationStore, runner: ResumeBus, logger=None, wait_registry=None
    ):
        self.store = store
        self.runner = runner
        self.logger = logger or log
        self.waits = wait_registry

    async def resume(
        self,
        run_id: str,
        node_id: str,
        token: str,
        payload: dict[str, Any],
    ) -> None:
        """Authorize one bearer token and resume its exact continuation.

        Examples:
            Resume an approval:
            ```python
            await router.resume("run-1", "approval", token, {"approved": True})
            ```
            Reject a mismatched route:
            ```python
            await router.resume("other-run", "approval", token, {})
            ```

        Args:
            run_id: Exact run identity asserted by the external route.
            node_id: Exact node identity asserted by the external route.
            token: Raw external bearer token.
            payload: Incoming resume payload.

        Returns:
            None: Delivery and terminal persistence are complete.

        Notes:
            This is the only bearer-validation boundary; downstream delivery is tokenless.
        """
        continuation = await self.store.resolve_token(token)
        if (
            continuation is None
            or continuation.run_id != run_id
            or continuation.node_id != node_id
            or continuation.closed
        ):
            self.logger.error("Invalid continuation or token for %s/%s", run_id, node_id)
            raise PermissionError("Invalid continuation or token")
        await self.resume_continuation(continuation, payload)

    async def resume_continuation(
        self,
        continuation: Continuation,
        payload: dict[str, Any],
    ) -> None:
        """Resume one trusted tokenless continuation identity.

        Examples:
            Deliver an indexed interaction:
            ```python
            await router.resume_continuation(resolved.continuation, payload)
            ```
            Deliver a due timer:
            ```python
            await router.resume_continuation(candidate, timer_payload)
            ```

        Args:
            continuation: Exact waiting record selected by trusted internal code.
            payload: Incoming or synthesized resume payload.

        Returns:
            None: Cooperative or scheduler delivery is complete.

        Notes:
            This method performs no token lookup and never weakens external authorization.
        """
        if continuation.closed:
            raise PermissionError("Invalid continuation or token")
        incoming = payload or {}
        if continuation.resume_schema:
            try:
                validate(instance=incoming, schema=continuation.resume_schema)
            except ValidationError as exc:
                self.logger.error("Resume payload validation error: %s", exc.message)
                raise ValueError(f"Invalid resume payload: {exc.message}") from exc

        full_payload: dict[str, Any] = {
            **(continuation.payload or {}),
            **incoming,
        }
        wait_id = continuation.continuation_id
        if self.waits and wait_id in getattr(self.waits, "_futs", {}):
            self.waits.resolve(wait_id, full_payload)
            await self.store.close(
                continuation,
                status=ContinuationStatus.RESUMED,
                closed_at=datetime.now(UTC),
            )
            self.logger.info(
                "Resolved cooperative wait for %s/%s", continuation.run_id, continuation.node_id
            )
            return

        await self.runner.enqueue_resume(continuation=continuation, payload=full_payload)
