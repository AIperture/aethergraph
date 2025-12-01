# tests/services/channel/test_web_ui_adapter.py
from __future__ import annotations

from typing import Any

import pytest

from aethergraph.contracts.services.channel import OutEvent
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.plugins.channel.adapters.webui import WebUIChannelAdapter
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.continuations.continuation import Correlator

# ---- Fakes / test doubles ----


class FakeEventLog(EventLog):
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    async def append(self, evt: dict) -> None:
        self.rows.append(evt)

    async def query(self, *, scope_id: str | None = None, since=None, until=None):
        # Very simple filter on scope_id; ignore time for now
        if scope_id is None:
            return list(self.rows)
        return [r for r in self.rows if r.get("scope_id") == scope_id]


class FakeStore:
    def __init__(self):
        self.bound: list[tuple[str, Correlator]] = []

    async def bind_correlator(self, token: str, corr: Correlator) -> None:
        self.bound.append((token, corr))


# ---- Tests ----


@pytest.mark.asyncio
async def test_web_ui_adapter_appends_event_with_run_scope():
    log = FakeEventLog()
    adapter = WebUIChannelAdapter(event_log=log)

    evt = OutEvent(
        type="agent.message",
        channel="ui:run/run-123",
        text="Hello from graph",
        meta={"foo": "bar"},
    )

    res = await adapter.send(evt)

    # Check result
    assert res is not None
    assert res["run_id"] == "run-123"

    corr = res.get("correlator")
    assert isinstance(corr, Correlator)
    assert corr.scheme == "ui"
    assert corr.channel == "ui:run/run-123"
    assert corr.thread == ""

    # Check event_log row
    assert len(log.rows) == 1
    row = log.rows[0]
    assert row["run_id"] == "run-123"
    assert row["scope_id"] == "run-ui:run-123"
    assert row["channel_key"] == "ui:run/run-123"
    assert row["type"] == "agent.message"
    assert row["text"] == "Hello from graph"
    assert row["meta"] == {"foo": "bar"}
    # ts, buttons, file exist but we only sanity-check types
    assert isinstance(row["ts"], float)
    assert isinstance(row["buttons"], list)
    assert row["file"] is None


@pytest.mark.asyncio
async def test_web_ui_adapter_binds_correlator_via_channel_bus():
    log = FakeEventLog()
    adapter = WebUIChannelAdapter(event_log=log)
    store = FakeStore()

    bus = ChannelBus(
        adapters={"ui": adapter},
        default_channel="ui:run/run-456",
        channel_aliases=None,
        store=store,
    )

    # This event simulates a session.need_input raised by notify()
    evt = OutEvent(
        type="session.need_input",
        channel="ui:run/run-456",
        text="Please reply",
        meta={"token": "token-123"},  # required for binding
    )

    # publish() will call adapter.send() and then _bind_correlator_if_any
    res = await bus.publish(evt)

    assert res is not None
    # Store should have exactly one bound correlator
    assert len(store.bound) == 1
    token, corr = store.bound[0]
    assert token == "token-123"
    assert isinstance(corr, Correlator)
    assert corr.scheme == "ui"
    assert corr.channel == "ui:run/run-456"
    # thread must be "" so ChannelIngress fallback matches it
    assert corr.thread == ""
