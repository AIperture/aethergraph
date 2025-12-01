# tests/api/test_run_channel_incoming.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# Import the router that contains /runs/{run_id}/channel/incoming
from aethergraph.plugins.channel.routes.webui_routes import router as channel_ui_router
from aethergraph.services.channel.ingress import ChannelIngress
from aethergraph.services.continuations.continuation import Continuation, Correlator

# ---- Fakes / test doubles ----


class FakeArtifacts:
    async def plan_staging_path(self, planned_ext: str | None = None) -> str:
        # Not used in this test (no files), but must exist
        return "/tmp/dummy"

    async def save_file(self, **kwargs):
        # Just return an object with a uri attribute
        @dataclass
        class Art:
            uri: str = "artifact://dummy"

        return Art()


class FakeKVHot:
    async def list_append_unique(self, key: str, items: list[dict[str, Any]], id_key: str) -> None:
        # We don't assert on this; just a stub
        self.last_call = (key, items, id_key)


class FakeContinuationStore:
    def __init__(self, cont: Continuation | None):
        self._cont = cont
        self.last_corr: Correlator | None = None

    async def find_by_correlator(self, corr: Correlator) -> Continuation | None:
        # Record for inspection; always return the same continuation in this test
        self.last_corr = corr
        return self._cont


class FakeResumeRouter:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def resume(self, *, run_id: str, node_id: str, token: str, payload: dict[str, Any]):
        self.calls.append(
            {
                "run_id": run_id,
                "node_id": node_id,
                "token": token,
                "payload": payload,
            }
        )


class FakeLogger:
    def __getattr__(self, name: str):
        # info/debug/warning... just swallow
        def _log(msg, *args, **kwargs):
            pass

        return _log

    def for_channel(self):
        return self


class FakeContainer:
    def __init__(self, cont: Continuation):
        self.artifacts = FakeArtifacts()
        self.kv_hot = FakeKVHot()
        self.cont_store = FakeContinuationStore(cont=cont)
        self.resume_router = FakeResumeRouter()
        self.logger = FakeLogger()
        # channel_ingress will be set after ChannelIngress is constructed


# ---- Helper to build app ----


def build_app_with_container(container: FakeContainer) -> FastAPI:
    app = FastAPI()
    app.state.container = container
    # Wire ChannelIngress
    container.channel_ingress = ChannelIngress(container=container, logger=container.logger)

    app.include_router(channel_ui_router, prefix="/api/v1")
    return app


# ---- Tests ----


@pytest.mark.asyncio
async def test_run_channel_incoming_resumes_continuation():
    # Fake Continuation waiting on user_input
    cont = Continuation(
        run_id="run-xyz",
        node_id="node-abc",
        token="tok-123",
        kind="user_input",
        channel="ui:run/run-xyz",
        prompt="Say something",
    )

    container = FakeContainer(cont=cont)
    app = build_app_with_container(container)
    client = TestClient(app)

    # Simulate a UI message to /runs/{run_id}/channel/incoming
    resp = client.post(
        "/api/v1/runs/run-xyz/channel/incoming",
        json={
            "text": "hello from ui",
            "meta": {"foo": "bar"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["resumed"] is True

    # Check that resume_router.resume was called with expected payload
    rr = container.resume_router
    assert len(rr.calls) == 1
    call = rr.calls[0]
    assert call["run_id"] == "run-xyz"
    assert call["node_id"] == "node-abc"
    assert call["token"] == "tok-123"

    payload = call["payload"]
    # For kind == "user_input", ChannelIngress builds payload with "text"
    assert payload["text"] == "hello from ui"
    assert payload["channel_key"] == "ui:chan/run/run-xyz" or payload["channel_key"].startswith(
        "ui:"
    )
    # meta should be preserved
    assert payload["meta"] == {"foo": "bar"}
