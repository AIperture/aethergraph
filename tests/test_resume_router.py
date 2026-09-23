"""Cooperative answers commit before they become visible to running work."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.continuations.continuation import ContinuationDraft, ContinuationStatus
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.waits.wait_registry import WaitRegistry


@pytest.fixture(params=["memory", "canonical"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryContinuationStore(secret=b"resume-test")
        return
    container = build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )
    try:
        yield container.cont_store
    finally:
        await container.run_manager.close()
        await container.close_storage()


async def _waiting(store):
    created = await store.create(
        ContinuationDraft(
            run_id="run",
            node_id="node",
            session_id="session",
            kind="user_input",
            prompt="Material?",
            payload={"prompt": "Material?", "_channel_wait_kind": "user_input"},
        )
    )
    waits = WaitRegistry()
    future = waits.register(created.continuation_id)
    runner = SimpleNamespace(enqueue_resume=AsyncMock())
    return (
        created.record,
        waits,
        future,
        ResumeRouter(store=store, runner=runner, wait_registry=waits),
    )


@pytest.mark.asyncio
async def test_storage_failure_does_not_release_cooperative_waiter(store, monkeypatch, caplog):
    continuation, waits, future, router = await _waiting(store)
    try:
        with monkeypatch.context() as patch:
            failure = AsyncMock(side_effect=OSError("storage unavailable"))
            patch.setattr(store, "close", failure)
            patch.setattr(store, "update", failure)
            with pytest.raises(OSError, match="storage unavailable"):
                await router.resume_continuation(continuation, {"text": "Titanium"})
        await asyncio.sleep(0)
        assert not future.done()
        assert "Cooperative response persistence failed" in caplog.text
        assert "Titanium" not in caplog.text
        assert (
            await store.get_by_id("run", "node", continuation.continuation_id)
        ).status is ContinuationStatus.WAITING
        await router.resume_continuation(continuation, {"text": "Titanium"})
        assert (await asyncio.wait_for(future, 1))["text"] == "Titanium"
        router.runner.enqueue_resume.assert_not_awaited()
    finally:
        waits.shutdown()


@pytest.mark.asyncio
async def test_cooperative_answer_is_retained_before_waiter_observes_it(store):
    continuation, waits, future, router = await _waiting(store)
    try:
        await router.resume_continuation(continuation, {"text": "Titanium"})
        answer = await asyncio.wait_for(future, 1)
        stored = await store.get_by_id("run", "node", continuation.continuation_id)
        assert stored.status is ContinuationStatus.RESUMED
        assert stored.payload == answer
        assert stored.payload["text"] == "Titanium"
        assert stored.payload["prompt"] == "Material?"
        with pytest.raises(PermissionError, match="stale"):
            await router.resume_continuation(continuation, {"text": "Aluminum"})
        router.runner.enqueue_resume.assert_not_awaited()
    finally:
        waits.shutdown()


@pytest.mark.asyncio
async def test_competing_answers_commit_one_response_without_a_second_delivery(store, monkeypatch):
    continuation, waits, future, router = await _waiting(store)
    original_update = store.update
    admitted = asyncio.Event()
    count = 0

    async def concurrent_update(record, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            admitted.set()
        await asyncio.wait_for(admitted.wait(), 1)
        return await original_update(record, **kwargs)

    monkeypatch.setattr(store, "update", concurrent_update)
    try:
        outcomes = await asyncio.gather(
            router.resume_continuation(continuation, {"text": "Titanium"}),
            router.resume_continuation(continuation, {"text": "Aluminum"}),
            return_exceptions=True,
        )
        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, Exception) for outcome in outcomes) == 1
        answer = await asyncio.wait_for(future, 1)
        stored = await store.get_by_id("run", "node", continuation.continuation_id)
        assert stored.payload == answer
        assert not waits.has(continuation.continuation_id)
        router.runner.enqueue_resume.assert_not_awaited()
    finally:
        waits.shutdown()


@pytest.mark.asyncio
async def test_answer_committed_during_waiter_cancellation_is_not_queued_again(store, monkeypatch):
    continuation, waits, future, router = await _waiting(store)
    original_update = store.update
    committed = asyncio.Event()
    release = asyncio.Event()

    async def delayed_return(record, **kwargs):
        stored = await original_update(record, **kwargs)
        committed.set()
        await release.wait()
        return stored

    monkeypatch.setattr(store, "update", delayed_return)
    response = asyncio.create_task(router.resume_continuation(continuation, {"text": "Titanium"}))
    try:
        await asyncio.wait_for(committed.wait(), 1)
        future.cancel()
        waits.cancel(continuation.continuation_id)
        release.set()
        await asyncio.wait_for(response, 1)
        assert not waits.has(continuation.continuation_id)
        stored = await store.get_by_id("run", "node", continuation.continuation_id)
        assert stored.status is ContinuationStatus.RESUMED
        assert stored.payload["text"] == "Titanium"
        router.runner.enqueue_resume.assert_not_awaited()
    finally:
        release.set()
        waits.shutdown()


@pytest.mark.asyncio
async def test_registry_preserves_explicit_early_response_and_respects_cancel_before_callback():
    waits = WaitRegistry()
    assert not waits.resolve("early", {"text": "ready"})
    early = waits.register("early")
    assert await asyncio.wait_for(early, 1) == {"text": "ready"}
    cancelled = waits.register("cancelled")
    assert waits.resolve("cancelled", {"text": "too late"}, cache_if_missing=False)
    cancelled.cancel()
    await asyncio.sleep(0)
    assert cancelled.cancelled()
    assert not waits.has("cancelled")
    waits.shutdown()
