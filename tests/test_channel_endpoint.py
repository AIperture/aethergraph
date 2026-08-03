from fastapi import FastAPI
import httpx
import pytest

from aethergraph.server.http.channel_http_routes import router as channel_router
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.channel.ingress import ChannelIngress
from aethergraph.services.channel.queue_adapter import QueueChannelAdapter
from aethergraph.services.continuations.continuation import Continuation  # or your real type

# ----- tiny in-memory fakes -----


class InMemoryKVHot:
    def __init__(self):
        self._lists = {}

    async def list_append(self, key, items):
        self._lists.setdefault(key, []).extend(items)

    async def list_get(self, key):
        return list(self._lists.get(key, []))

    async def list_append_unique(self, key, items, id_key="id"):
        seen = {x[id_key] for x in self._lists.get(key, [])}
        new = [x for x in items if x[id_key] not in seen]
        await self.list_append(key, new)


class InMemoryContStore:
    def __init__(self):
        self.by_token = {}
        self.by_corr_key = {}  # (scheme, channel, thread) -> token

    async def save(self, cont: Continuation):
        self.by_token[cont.token] = cont

    async def bind_correlator(self, token, corr):
        key = (corr.scheme, corr.channel, corr.thread or "")
        self.by_corr_key[key] = token

    async def find_by_correlator(self, corr):
        key = (corr.scheme, corr.channel, corr.thread or "")
        token = self.by_corr_key.get(key)
        return self.by_token.get(token)


class InMemoryResumeRouter:
    def __init__(self):
        self.calls = []

    async def resume(self, run_id, node_id, token, payload):
        self.calls.append((run_id, node_id, token, payload))


class ChannelTestContainer:
    def __init__(self):
        self.kv_hot = InMemoryKVHot()
        self.cont_store = InMemoryContStore()
        self.resume_router = InMemoryResumeRouter()
        self.logger = None

        # channel bus + adapter
        self.channel_bus = ChannelBus(
            adapters={},
            resume_router=self.resume_router,
            store=self.cont_store,
            logger=self.logger,
        )
        # generic queue adapter for ext:*
        self.queue_adapter = QueueChannelAdapter(self, scheme="ext")
        self.channel_bus.register_adapter("ext", self.queue_adapter)

        # ingress for generic channels
        self.channel_ingress = ChannelIngress(container=self, logger=self.logger)


def build_app(container: ChannelTestContainer) -> FastAPI:
    app = FastAPI()
    app.state.container = container
    app.state.settings = type("S", (), {})()  # minimal stub if needed
    app.include_router(channel_router, prefix="")  # /channel/incoming, /channel/resume
    return app


# ----- the actual test -----


@pytest.mark.anyio
async def test_channel_http_roundtrip():
    container = ChannelTestContainer()
    app = build_app(container)

    # Create a waiting continuation manually
    cont = Continuation(
        run_id="run-1",
        node_id="node-1",
        token="tok-1",
        kind="user_input",
        channel="ext:chan/user-123",
        prompt="Say something",
        payload={"_interaction_id": "interaction-endpoint-1"},
    )
    await container.cont_store.save(cont)

    # Notify via ChannelBus: this should send an OutEvent and bind correlator
    _ = await container.channel_bus.notify(cont)
    # If QueueChannelAdapter doesn't return a correlator yet, res may be {}; that's fine.
    # The important part is that an outbox event exists and cont_store has a correlator

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/channel/incoming",
            json={
                "scheme": "ext",
                "channel_id": "user-123",
                "text": "hello from outside",
                "meta": {"foo": "bar"},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True

    # Check that resume_router was called with expected payload
    assert len(container.resume_router.calls) == 1
    run_id, node_id, token, payload = container.resume_router.calls[0]
    assert run_id == "run-1"
    assert node_id == "node-1"
    assert token == "tok-1"
    assert payload["text"] == "hello from outside"
    assert payload["meta"]["foo"] == "bar"

    # print("Test passed: channel HTTP roundtrip works as expected.")


@pytest.mark.anyio
async def test_channel_http_accepts_canonical_attachments():
    container = ChannelTestContainer()
    app = build_app(container)

    cont = Continuation(
        run_id="run-1",
        node_id="node-1",
        token="tok-1",
        kind="user_input",
        channel="ext:chan/user-123",
        prompt="Send context",
        payload={"_interaction_id": "interaction-endpoint-2"},
    )
    await container.cont_store.save(cont)
    await container.channel_bus.notify(cont)

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/channel/incoming",
            json={
                "scheme": "ext",
                "channel_id": "user-123",
                "text": "use this",
                "attachments": [
                    {
                        "kind": "artifact",
                        "source": "external_api",
                        "artifact_id": "art-1",
                        "name": "input.txt",
                        "mime": "text/plain",
                    }
                ],
            },
        )
        assert r.status_code == 200

    assert len(container.resume_router.calls) == 1
    _, _, _, payload = container.resume_router.calls[0]
    assert payload["attachments"][0]["artifact_id"] == "art-1"
    assert payload["attachments"][0]["mime"] == "text/plain"
    assert payload["meta"]["attachments"][0]["artifact_id"] == "art-1"


@pytest.mark.anyio
async def test_channel_prompt_preserves_interaction_rendering_hints():
    container = ChannelTestContainer()
    cont = Continuation(
        run_id="run-1",
        node_id="node-1",
        token="tok-files",
        kind="user_files",
        channel="ext:chan/user-123",
        prompt="Upload evidence",
        payload={
            "prompt": "Upload evidence",
            "accept": ["image/png", ".pdf"],
            "multiple": False,
            "_interaction_id": "interaction-endpoint-files",
        },
    )

    await container.channel_bus.notify(cont)

    events = await container.kv_hot.list_get("outbox://ext:chan/user-123")
    assert len(events) == 1
    assert events[0]["type"] == "session.need_input"
    assert events[0]["meta"]["interaction_kind"] == "user_files"
    assert events[0]["meta"]["accept"] == ["image/png", ".pdf"]
    assert events[0]["meta"]["multiple"] is False
