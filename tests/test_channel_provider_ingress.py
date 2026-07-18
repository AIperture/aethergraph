from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.contracts.services.channel import Button, OutEvent
from aethergraph.plugins.channel.adapters.slack import SlackChannelAdapter
from aethergraph.plugins.channel.adapters.telegram import TelegramChannelAdapter
from aethergraph.plugins.channel.utils import slack_utils, telegram_utils
from aethergraph.services.channel.ingress import ChannelIngress
from aethergraph.services.continuations.continuation import Continuation


class _EventLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


class _ContinuationStore:
    def __init__(self, continuation: Continuation | None) -> None:
        self.continuation = continuation

    async def find_by_correlator(self, *, corr) -> Continuation | None:
        return self.continuation

    async def list_waits(self) -> list[dict[str, Any]]:
        if self.continuation is None:
            return []
        return [self.continuation.to_dict()]

    async def delete(self, run_id: str, node_id: str) -> None:
        self.continuation = None


class _ResumeRouter:
    def __init__(self, store: _ContinuationStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []

    async def resume(
        self,
        *,
        run_id: str,
        node_id: str,
        token: str,
        payload: dict[str, Any],
    ) -> None:
        self.calls.append(
            {
                "run_id": run_id,
                "node_id": node_id,
                "token": token,
                "payload": payload,
            }
        )
        await self.store.delete(run_id, node_id)


class _Logger:
    def for_run(self):
        return self

    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _Container:
    def __init__(self, continuation: Continuation | None, *, default_agent_id: str | None = None):
        self.cont_store = _ContinuationStore(continuation)
        self.resume_router = _ResumeRouter(self.cont_store)
        self.eventlog = _EventLog()
        self.kv_hot = None
        self.logger = _Logger()
        self.wait_registry = None
        self.sched_registry = None
        self.channels = SimpleNamespace(adapters={})
        self.settings = SimpleNamespace(telegram=SimpleNamespace(default_agent_id=default_agent_id))
        self.channel_ingress = ChannelIngress(container=self, logger=self.logger)


def _slack_settings(default_agent_id: str | None = None):
    return SimpleNamespace(
        slack=SimpleNamespace(
            bot_token=None,
            default_agent_id=default_agent_id,
        )
    )


@pytest.mark.asyncio
async def test_slack_shared_message_handler_records_once_and_resumes():
    continuation = Continuation(
        run_id="run-slack",
        node_id="node-slack",
        token="slack-token",
        kind="user_input",
        prompt="Reply",
    )
    container = _Container(continuation)

    await slack_utils.handle_slack_events_common(
        container,
        _slack_settings(),
        {
            "event_id": "Ev-1",
            "team_id": "T1",
            "event": {
                "type": "message",
                "channel": "C1",
                "user": "U1",
                "ts": "100.1",
                "text": "hello from Slack",
            },
        },
    )

    assert len(container.eventlog.rows) == 1
    row = container.eventlog.rows[0]
    assert row["kind"] == "channel_inbound"
    assert row["payload"]["text"] == "hello from Slack"
    assert row["payload"]["meta"]["slack"]["event_id"] == "Ev-1"
    assert len(container.resume_router.calls) == 1


@pytest.mark.asyncio
async def test_telegram_shared_message_handler_records_once_and_resumes():
    continuation = Continuation(
        run_id="run-tg",
        node_id="node-tg",
        token="telegram-token",
        kind="user_input",
        prompt="Reply",
    )
    container = _Container(continuation)

    await telegram_utils._process_update(
        container,
        {
            "update_id": 123,
            "message": {
                "message_id": 7,
                "date": 10,
                "from": {"id": 42, "is_bot": False},
                "chat": {"id": 99},
                "text": "hello from Telegram",
            },
        },
        token="",
    )

    assert len(container.eventlog.rows) == 1
    row = container.eventlog.rows[0]
    assert row["kind"] == "channel_inbound"
    assert row["payload"]["text"] == "hello from Telegram"
    assert row["payload"]["meta"]["telegram"]["update_id"] == 123
    assert len(container.resume_router.calls) == 1


@pytest.mark.asyncio
async def test_shared_message_handlers_dispatch_only_when_not_resumed(monkeypatch):
    dispatches: list[dict[str, Any]] = []

    async def _dispatch(**kwargs) -> str:
        dispatches.append(kwargs)
        return "new-run"

    monkeypatch.setattr(slack_utils, "dispatch_channel_turn_run", _dispatch)
    monkeypatch.setattr(telegram_utils, "dispatch_channel_turn_run", _dispatch)

    slack_container = _Container(None, default_agent_id="agent-1")
    await slack_utils.handle_slack_events_common(
        slack_container,
        _slack_settings(default_agent_id="agent-1"),
        {
            "event_id": "Ev-root",
            "team_id": "T1",
            "event": {
                "type": "message",
                "channel": "C1",
                "user": "U1",
                "ts": "100.2",
                "text": "start Slack turn",
            },
        },
    )

    telegram_container = _Container(None, default_agent_id="agent-1")
    await telegram_utils._process_update(
        telegram_container,
        {
            "update_id": 124,
            "message": {
                "message_id": 8,
                "from": {"id": 42, "is_bot": False},
                "chat": {"id": 99},
                "text": "start Telegram turn",
            },
        },
        token="",
    )

    assert len(slack_container.eventlog.rows) == 1
    assert len(telegram_container.eventlog.rows) == 1
    assert [call["text"] for call in dispatches] == [
        "start Slack turn",
        "start Telegram turn",
    ]


@pytest.mark.asyncio
async def test_slack_callback_uses_exact_token_and_replay_records_nothing():
    continuation = Continuation(
        run_id="run-slack",
        node_id="node-slack",
        token="slack-token",
        kind="choice",
        prompt={
            "title": "Choose",
            "choices": [
                {"id": "ship", "label": "Ship It"},
                {"id": "revise", "label": "Revise"},
            ],
        },
    )
    container = _Container(continuation)
    payload = {
        "trigger_id": "trigger-1",
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "channel": {"id": "C1"},
        "message": {"ts": "100.3"},
        "actions": [
            {
                "action_id": "ag_button_0",
                "action_ts": "100.4",
                "value": (
                    '{"choice":"ship","choice_label":"Ship It",'
                    '"run_id":"run-slack","node_id":"node-slack",'
                    '"token":"slack-token"}'
                ),
            }
        ],
    }

    await slack_utils.handle_slack_interactive_common(container, payload)
    await slack_utils.handle_slack_interactive_common(container, payload)

    assert len(container.eventlog.rows) == 1
    assert len(container.resume_router.calls) == 1
    row = container.eventlog.rows[0]
    assert row["payload"]["choice"] == "ship"
    assert row["payload"]["text"] == "Ship It"
    assert row["payload"]["meta"]["slack"]["action_id"] == "ag_button_0"


@pytest.mark.asyncio
async def test_telegram_callback_resolves_alias_and_replay_records_nothing():
    token = "123456789012345678901234-rest-of-token"
    continuation = Continuation(
        run_id="run-tg",
        node_id="node-tg",
        token=token,
        kind="choice",
        prompt={
            "title": "Choose",
            "choices": [
                {"id": "ship", "label": "Ship It"},
                {"id": "revise", "label": "Revise"},
            ],
        },
    )
    container = _Container(continuation)
    payload = {
        "update_id": 125,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 42},
            "data": "i=2|k=123456789012345678901234",
            "message": {
                "message_id": 9,
                "chat": {"id": 99},
            },
        },
    }

    await telegram_utils._process_update(container, payload, token="")
    await telegram_utils._process_update(container, payload, token="")

    assert len(container.eventlog.rows) == 1
    assert len(container.resume_router.calls) == 1
    row = container.eventlog.rows[0]
    assert row["payload"]["choice"] == "revise"
    assert row["payload"]["text"] == "Revise"
    assert row["payload"]["meta"]["telegram"]["callback_id"] == "callback-1"


@pytest.mark.asyncio
async def test_slack_buttons_carry_choice_label_and_exact_token():
    posted: list[dict[str, Any]] = []

    class _SlackClient:
        async def chat_postMessage(self, **kwargs):
            posted.append(kwargs)
            return {"ts": "100.5"}

    adapter = object.__new__(SlackChannelAdapter)
    adapter.client = _SlackClient()
    adapter._first_ts_by_chan = {}
    await adapter.send(
        OutEvent(
            type="link.buttons",
            channel="slack:team/T1:chan/C1:thread/100.1",
            text="Choose",
            buttons=[Button(label="Ship It", value="ship")],
            meta={
                "run_id": "run-slack",
                "node_id": "node-slack",
                "token": "exact-token",
            },
        )
    )

    value = json.loads(posted[0]["blocks"][1]["elements"][0]["value"])
    assert value == {
        "choice": "ship",
        "choice_label": "Ship It",
        "run_id": "run-slack",
        "node_id": "node-slack",
        "token": "exact-token",
    }


@pytest.mark.asyncio
async def test_telegram_buttons_keep_compact_resume_alias_for_four_choices():
    sent: list[dict[str, Any]] = []

    async def _api(method: str, **kwargs):
        sent.append({"method": method, **kwargs})
        return {"result": {"message_id": 10}}

    adapter = object.__new__(TelegramChannelAdapter)
    adapter._msg_id_cache = {}
    adapter._api = _api
    await adapter.send(
        OutEvent(
            type="link.buttons",
            channel="tg:chat/99",
            text="Choose",
            buttons=[
                Button(label="One", value="one"),
                Button(label="Two", value="two"),
                Button(label="Three", value="three"),
                Button(label="Four", value="four"),
            ],
            meta={"resume_key": "123456789012345678901234"},
        )
    )

    keyboard = sent[0]["reply_markup"]["inline_keyboard"]
    callback_data = [row[0]["callback_data"] for row in keyboard]
    assert callback_data == [
        "i=1|k=123456789012345678901234",
        "i=2|k=123456789012345678901234",
        "i=3|k=123456789012345678901234",
        "i=4|k=123456789012345678901234",
    ]
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_data)
