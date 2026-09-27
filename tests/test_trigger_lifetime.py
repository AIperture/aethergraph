"""Trigger shutdown owns in-flight claims and cannot lose an immediate stop."""

import asyncio
from types import SimpleNamespace

import pytest

from aethergraph.services.triggers.engine import TriggerEngine


class BlockingStore:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.scans = 0

    async def claim_due(self, *args, **kwargs):
        self.scans += 1
        self.entered.set()
        await self.release.wait()
        return []


@pytest.mark.asyncio
async def test_immediate_stop_cannot_be_overwritten_by_starting_loop():
    store = BlockingStore()
    engine = TriggerEngine(store=store, run_manager=SimpleNamespace())
    await engine.start()
    await engine.stop()
    assert store.scans == 0


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_joins_the_active_claim_scan():
    store = BlockingStore()
    engine = TriggerEngine(store=store, run_manager=SimpleNamespace())
    await engine.start()
    await store.entered.wait()
    await engine.start()
    stopped = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not stopped.done()
    assert store.scans == 1
    store.release.set()
    await stopped
    await engine.stop()
