import asyncio
import time
from typing import Any

from fastapi import FastAPI
import httpx
import pytest

from aethergraph import graphify, tool
from aethergraph.core.execution.global_scheduler import RunSettings
from aethergraph.core.runtime.graph_runner import _build_env, _resolve_graph_outputs
from aethergraph.core.tools.waitable import DualStageTool, WaitSpec
from aethergraph.server.clients.channel_client import ChannelClient
from aethergraph.server.http.channel_http_routes import router as channel_router


def now_ms() -> str:
    return f"{(time.perf_counter() * 1000):.0f}ms"


@tool(outputs=["result"])
async def slow_a(x: int) -> dict:
    print(f"[{now_ms()}] slow_a start x={x}")
    await asyncio.sleep(0.3)
    print(f"[{now_ms()}] slow_a end   x={x}")
    return {"result": x * 2}


@tool(outputs=["decision"])
class ApproveWait(DualStageTool):
    """Ask for a human decision, then continue with the choice."""

    outputs = ["decision"]

    async def setup(
        self, *, prompt: str = "Approve this run?", options: list[str] = None, **kwargs
    ):
        # First stage: request a wait with schema and channel prompt
        opts = options or ["Yes", "No", "Maybe"]
        return WaitSpec(
            kind="approval",
            prompt={"text": prompt, "options": opts},
            resume_schema={
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": opts}},
                "required": ["choice"],
            },
            # channel resolved by normalize_wait_spec via NodeContext if omitted
        )

    async def on_resume(self, resume: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Second stage: produce the final outputs based on the resume payload
        choice = resume.get("choice", "No")
        print(f"[{now_ms()}] on_resume received: {choice}")
        return {"decision": choice}


# combine result with decision
@tool(outputs=["final_result", "decision"])
async def combine(value: int, decision: str) -> dict:
    return {"final_result": value, "decision": decision}


@graphify(
    name="wait_demo_gf",
    inputs=["x"],
    outputs=["final_result", "decision"],
)
def wait_demo(x: int = 5):
    a = slow_a(x=x)  # quick compute
    w = ApproveWait(prompt="Proceed with doubling result?")  # WAITING_HUMAN until resume
    c = combine(value=a.result, decision=w.decision)
    return {"final_result": c.final_result, "decision": c.decision}


async def _setup_wait_demo_run(x: int = 7):
    """
    Helper to:
      - materialize the static graph
      - build env
      - submit to global scheduler
      - start run_until_complete in background

    Returns:
      task_graph, env, waiter_task
    """
    # 1) Materialize graph from graphify factory
    task_graph = wait_demo.build()

    # 2) Build env + schedulers
    env, retry_policy, max_conc = await _build_env(task_graph, inputs={"x": x})

    scheds = getattr(env, "schedulers", None) or getattr(
        getattr(env, "container", None), "schedulers", None
    )
    global_sched = scheds["global"]

    settings = RunSettings(
        max_concurrency=2,
        retry_policy=retry_policy,
        stop_on_first_error=True,
        skip_dependents_on_failure=True,
    )

    # 3) Submit run and start scheduler in background
    await global_sched.submit(run_id=env.run_id, graph=task_graph, env=env, settings=settings)
    waiter = asyncio.create_task(global_sched.run_until_complete(env.run_id))

    return task_graph, env, waiter


async def _wait_for_continuation(env, *, timeout_s: float = 5.0):
    """
    Poll the continuation store for a continuation created by ApproveWait.
    """
    resume_router = env.container.resume_router
    store = resume_router.store

    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        conts = await store.list_cont_by_run(env.run_id)
        if conts:
            # In more complex graphs, filter by kind == "approval"
            return conts[0]
        await asyncio.sleep(0.05)

    raise AssertionError("No continuation found; did ApproveWait request a wait?")


def _build_test_app(env) -> FastAPI:
    """
    Minimal FastAPI app wired to this env's container, exposing /channel/resume.
    """
    app = FastAPI()
    app.state.container = env.container
    app.include_router(channel_router, prefix="")  # /channel/incoming, /channel/resume
    return app


# ---------------------------------------------------------------------------
# Test 1: Use raw AsyncClient to hit /channel/resume directly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dualstage_resume_via_http_client():
    # 1) Set up run + scheduler
    task_graph, env, waiter = await _setup_wait_demo_run(x=7)

    # 2) Build ASGI app + transport
    app = _build_test_app(env)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 3) Wait until the DualStage tool has created a waiting continuation
        cont = await _wait_for_continuation(env)
        print(
            f"[TEST/http] Found continuation: run_id={cont.run_id} "
            f"node_id={cont.node_id} token={cont.token}"
        )

        # 4) Simulate user resuming via /channel/resume
        resp = await client.post(
            "/channel/resume",
            json={
                "run_id": cont.run_id,
                "node_id": cont.node_id,
                "token": cont.token,
                "payload": {"choice": "Yes"},  # must satisfy resume_schema
            },
        )
        print("[TEST/http] /channel/resume response:", resp.status_code, resp.text)
        assert resp.status_code == 200

        # 5) Wait for run completion
        await waiter

        # 6) Check final outputs
        result = _resolve_graph_outputs(task_graph, inputs={"x": 7}, env=env)
        print("[TEST/http] FINAL RESULT:", result)

        assert result["final_result"] == 7 * 2
        assert result["decision"] == "Yes"


# ---------------------------------------------------------------------------
# Test 2: Use ChannelClient.resume(...) (recommended external usage)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dualstage_resume_via_channel_client():
    # 1) Set up run + scheduler
    task_graph, env, waiter = await _setup_wait_demo_run(x=7)

    # 2) Build ASGI app + transport
    app = _build_test_app(env)
    transport = httpx.ASGITransport(app=app)

    async_client = httpx.AsyncClient(transport=transport, base_url="http://test")

    try:
        # 3) ChannelClient wired to the same base_url, using injected AsyncClient
        chan_client = ChannelClient(
            base_url="http://test",
            scheme="ext",  # arbitrary; resume doesn't actually use scheme/channel_id
            channel_id="user-123",
            http_client=async_client,
        )

        # 4) Wait until a continuation exists
        cont = await _wait_for_continuation(env)
        print(
            f"[TEST/client] Found continuation: run_id={cont.run_id} "
            f"node_id={cont.node_id} token={cont.token}"
        )

        # 5) Use ChannelClient.resume to simulate the human choice
        await chan_client.resume(
            run_id=cont.run_id,
            node_id=cont.node_id,
            token=cont.token,
            payload={"choice": "Yes"},
        )

        # 6) Wait for the run to finish
        await waiter

        # 7) Check final outputs
        result = _resolve_graph_outputs(task_graph, inputs={"x": 7}, env=env)
        print("[TEST/client] FINAL RESULT:", result)

        assert result["final_result"] == 7 * 2
        assert result["decision"] == "Yes"

    finally:
        await async_client.aclose()


if __name__ == "__main__":

    async def main():
        # Call tests sequentially on the SAME event loop
        await test_dualstage_resume_via_http_client()
        await test_dualstage_resume_via_channel_client()

    asyncio.run(main())
