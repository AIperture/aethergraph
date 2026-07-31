"""Container-scoped reactive rate-limit gate."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import time

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class ProviderRateGate:
    """Coordinate provider-advised waits across clients in one container."""

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """
        Construct an empty container-local gate.

        Examples:
            Build the production gate:
                ```python
                gate = ProviderRateGate()
                ```

            Inject deterministic test dependencies:
                ```python
                gate = ProviderRateGate(clock=lambda: 10.0, sleep=fake_sleep)
                ```

        Args:
            clock: Monotonic clock used for blocked-until deadlines.
            sleep: Cancellation-aware asynchronous sleep implementation.

        Returns:
            None: Initializes an empty keyed gate.

        Notes:
            The gate is intentionally process-local. It neither predicts token
            usage nor claims coordination with other containers.
        """

        self._clock = clock
        self._sleep = sleep
        self._blocked_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def defer(self, key: str, delay_s: float) -> float:
        """
        Extend a key's blocked interval without shortening an existing delay.

        Examples:
            Record a provider-advised delay:
                ```python
                blocked_until = await gate.defer("openai:model", 0.5)
                assert blocked_until >= 0.5
                ```

            Ignore a negative delay:
                ```python
                blocked_until = await gate.defer("openai:model", -1.0)
                assert blocked_until >= 0.0
                ```

        Args:
            key: Stable provider/model or explicit rate-limit group key.
            delay_s: Minimum delay from the current monotonic time.

        Returns:
            float: Effective monotonic blocked-until deadline.

        Notes:
            Concurrent callers only extend the deadline. They never replace a
            longer block with a shorter one.
        """

        proposed = self._clock() + max(0.0, float(delay_s))
        async with self._lock:
            current = self._blocked_until.get(key, 0.0)
            effective = max(current, proposed)
            self._blocked_until[key] = effective
            return effective

    async def wait(self, key: str) -> float:
        """
        Wait until the current blocked interval for a key has elapsed.

        Examples:
            Pass through an unblocked key:
                ```python
                waited = await gate.wait("ollama:local")
                assert waited == 0.0
                ```

            Share one deadline across callers:
                ```python
                await gate.defer("openai:model", 1.0)
                waited = await gate.wait("openai:model")
                assert waited >= 0.0
                ```

        Args:
            key: Stable provider/model or explicit rate-limit group key.

        Returns:
            float: Total requested wait duration for this invocation.

        Notes:
            Cancellation propagates from the injected sleep. No waiter is
            retained by the gate after cancellation.
        """

        total_wait_s = 0.0
        while True:
            async with self._lock:
                blocked_until = self._blocked_until.get(key, 0.0)
            remaining_s = blocked_until - self._clock()
            if remaining_s <= 0.0:
                async with self._lock:
                    if self._blocked_until.get(key, 0.0) <= self._clock():
                        self._blocked_until.pop(key, None)
                return total_wait_s
            total_wait_s += remaining_s
            await self._sleep(remaining_s)
