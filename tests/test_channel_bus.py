from __future__ import annotations

import pytest

from aethergraph.contracts.services.channel import Button, ChannelRoutingError, OutEvent
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.continuations.continuation import Continuation, Correlator


class _CapturingAdapter:
    capabilities = {"text"}

    def __init__(self) -> None:
        self.events: list[OutEvent] = []

    async def send(self, event: OutEvent) -> dict[str, str]:
        self.events.append(event)
        return {"delivery_id": "delivery-1"}


@pytest.mark.asyncio
async def test_publish_delivers_unsupported_event_unchanged() -> None:
    adapter = _CapturingAdapter()
    bus = ChannelBus(adapters={"test": adapter})
    event = OutEvent(
        type="session.need_approval",
        channel="test:conversation/one",
        text="Approve?",
        buttons=[Button(label="Approve", value="approve")],
    )

    result = await bus.publish(event)

    assert result == {"delivery_id": "delivery-1"}
    assert adapter.events == [event]
    assert adapter.events[0].type == "session.need_approval"
    assert adapter.events[0].buttons == [Button(label="Approve", value="approve")]


@pytest.mark.asyncio
async def test_publish_reports_structured_missing_adapter() -> None:
    bus = ChannelBus(adapters={"ui": _CapturingAdapter()})
    event = OutEvent(type="agent.message", channel="slack:team/T:chan/C", text="Done")

    with pytest.raises(ChannelRoutingError) as exc_info:
        await bus.publish(event)

    assert exc_info.value.code == "channel.adapter_not_found"
    assert exc_info.value.channel_key == "slack:team/T:chan/C"
    assert exc_info.value.known_prefixes == ("ui",)


@pytest.mark.asyncio
async def test_notify_exposes_only_public_interaction_identity() -> None:
    class _PushAdapter:
        capabilities = {"text", "buttons"}

        def __init__(self) -> None:
            self.events: list[OutEvent] = []

        async def send(self, event: OutEvent) -> dict[str, Correlator]:
            self.events.append(event)
            return {
                "correlator": Correlator(
                    scheme="slack",
                    channel=event.channel,
                    thread="1",
                    message="2",
                )
            }

    class _Store:
        def __init__(self) -> None:
            self.bindings: list[tuple[str, Correlator]] = []

        async def bind_correlator(self, *, token: str, corr: Correlator) -> None:
            self.bindings.append((token, corr))

    adapter = _PushAdapter()
    store = _Store()
    bus = ChannelBus(adapters={"slack": adapter}, store=store)
    continuation = Continuation(
        run_id="run-secret",
        node_id="node-secret",
        token="token-secret",
        kind="choice",
        channel="slack:team/T:chan/C",
        prompt={"title": "Choose", "choices": [{"id": "ship", "label": "Ship"}]},
        payload={"_interaction_id": "interaction-public-1"},
    )

    await bus.notify(continuation)

    event_meta = adapter.events[0].meta
    assert event_meta["interaction_id"] == "interaction-public-1"
    assert "token" not in event_meta
    assert "resume_key" not in event_meta
    assert "run_id" not in event_meta
    assert "node_id" not in event_meta
    assert store.bindings[0][0] == "token-secret"


@pytest.mark.asyncio
async def test_notify_rejects_continuation_without_public_interaction_identity() -> None:
    bus = ChannelBus(adapters={"test": _CapturingAdapter()})
    continuation = Continuation(
        run_id="run-1",
        node_id="node-1",
        token="token-1",
        kind="user_input",
        channel="test:conversation/one",
        prompt="Reply",
    )

    with pytest.raises(ValueError, match="public interaction identity"):
        await bus.notify(continuation)
