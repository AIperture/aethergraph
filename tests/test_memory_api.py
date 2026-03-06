from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.contracts.services.memory import Event
from aethergraph.core.runtime.runtime_services import install_services
from aethergraph.server.app_factory import create_app


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def app(tmp_path):
    cfg = load_settings()
    set_current_settings(cfg)
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    install_services(app.state.container)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.asyncio
async def _seed_events(app, timeline_id: str, *, memory_scope_id: str):
    container = app.state.container
    persistence = container.memory_factory.persistence

    base_ts = _utc_now() - timedelta(minutes=10)
    events = [
        Event(
            event_id="e1",
            ts=(base_ts + timedelta(minutes=1)).isoformat(),
            run_id="run-1",
            scope_id=memory_scope_id,
            user_id="local",
            org_id="local",
            session_id="sess-1",
            agent_id="agent-a",
            graph_id="graph-1",
            node_id="node-1",
            kind="chat_user",
            stage="user",
            text="hello world from the user",
            tags=["chat", "user"],
            data={"text": "hello world from the user"},
            metrics=None,
        ),
        Event(
            event_id="e2",
            ts=(base_ts + timedelta(minutes=2)).isoformat(),
            run_id="run-2",
            scope_id=memory_scope_id,
            user_id="local",
            org_id="local",
            session_id="sess-1",
            agent_id="agent-a",
            graph_id="graph-1",
            node_id="node-2",
            kind="chat_assistant",
            stage="assistant",
            text="assistant reply with some extra detail for truncation checks",
            tags=["chat", "assistant"],
            data={"text": "assistant reply with some extra detail for truncation checks"},
            metrics=None,
            severity=3,
            signal=0.7,
        ),
        Event(
            event_id="e3",
            ts=(base_ts + timedelta(minutes=3)).isoformat(),
            run_id="run-3",
            scope_id=memory_scope_id,
            user_id="local",
            org_id="local",
            session_id="sess-1",
            agent_id="agent-b",
            graph_id="graph-2",
            node_id="node-3",
            kind="tool_result",
            stage="tool",
            text=None,
            tags=["tool", "result"],
            data={"value": 42, "text": "tool output"},
            metrics={"latency_ms": 10.0},
            tool="search",
        ),
        Event(
            event_id="e4",
            ts=(base_ts + timedelta(minutes=4)).isoformat(),
            run_id="run-4",
            scope_id="run-4",
            user_id="local",
            org_id="local",
            session_id="sess-1",
            agent_id=None,
            graph_id=None,
            node_id=None,
            kind="checkpoint",
            stage="system",
            text=None,
            tags=["system"],
            data={"state": "saved"},
            metrics=None,
        ),
    ]

    for evt in events:
        await persistence.append_event(timeline_id, evt)


def test_list_memory_events_recent_first_and_pagination(app, client):
    timeline_id = "org:local|session:sess-1"
    asyncio.run(_seed_events(app, timeline_id, memory_scope_id="session:sess-1"))

    resp = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "limit": 2},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert [evt["event_id"] for evt in body["events"]] == ["e4", "e3"]
    assert body["next_cursor"] is not None
    assert body["events"][0]["snippet"] == '{"state": "saved"}'
    assert body["events"][1]["tool"] == "search"

    resp2 = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "limit": 2, "cursor": body["next_cursor"]},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert [evt["event_id"] for evt in body2["events"]] == ["e2", "e1"]
    assert body2["next_cursor"] is None


def test_list_memory_events_filters_session_agent_and_tags(app, client):
    timeline_id = "org:local|session:sess-1"
    asyncio.run(_seed_events(app, timeline_id, memory_scope_id="session:sess-1"))

    resp = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "agent_id": "agent-a"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [evt["event_id"] for evt in body["events"]] == ["e2", "e1"]
    assert all(evt["agent_id"] == "agent-a" for evt in body["events"])

    resp2 = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "tags": "tool"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert [evt["event_id"] for evt in body2["events"]] == ["e3"]
    assert body2["events"][0]["snippet"] == "tool output"


def test_list_memory_events_scope_fallback_and_schema_fields(app, client):
    timeline_id = "org:local|session:sess-1"
    asyncio.run(_seed_events(app, timeline_id, memory_scope_id="session:sess-1"))

    resp = client.get(
        "/api/v1/memory/events",
        params={"scope_id": timeline_id, "kinds": "chat_assistant"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["events"]) == 1
    event = body["events"][0]
    assert event["event_id"] == "e2"
    assert event["scope_id"] == "session:sess-1"
    assert event["session_id"] == "sess-1"
    assert event["run_id"] == "run-2"
    assert event["graph_id"] == "graph-1"
    assert event["node_id"] == "node-2"
    assert event["stage"] == "assistant"
    assert event["severity"] == 3
    assert event["signal"] == 0.7
    assert event["text"].startswith("assistant reply")


def test_list_memory_events_isolates_tenant_identity(app, client):
    container = app.state.container
    persistence = container.memory_factory.persistence
    timeline_id = "org:shared|session:sess-tenant"
    now = _utc_now().isoformat()

    asyncio.run(
        persistence.append_event(
            timeline_id,
            Event(
                event_id="tenant-a",
                ts=now,
                run_id="run-a",
                scope_id="session:sess-tenant",
                user_id="user-a",
                org_id="org-a",
                session_id="sess-tenant",
                kind="user_msg",
                text="from tenant a",
            ),
        )
    )
    asyncio.run(
        persistence.append_event(
            timeline_id,
            Event(
                event_id="tenant-b",
                ts=now,
                run_id="run-b",
                scope_id="session:sess-tenant",
                user_id="user-b",
                org_id="org-b",
                session_id="sess-tenant",
                kind="user_msg",
                text="from tenant b",
            ),
        )
    )

    resp = client.get(
        "/api/v1/memory/events",
        params={"scope_id": timeline_id},
        headers={"X-User-ID": "user-a", "X-Org-ID": "org-a"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [evt["event_id"] for evt in body["events"]] == ["tenant-a"]


def test_list_memory_summaries_filters_in_persistence(app, client):
    container = app.state.container
    persistence = container.memory_factory.persistence

    asyncio.run(
        persistence.save_json(
            "file://mem/session:sess-1/summaries/session/2026-03-06T00-00-00Z.json",
            {
                "summary_doc_id": "summary-local",
                "scope_id": "session:sess-1",
                "summary_tag": "session",
                "summary": "local summary",
                "ts": "2026-03-06T00:00:00+00:00",
                "org_id": "local",
                "user_id": "local",
            },
        )
    )
    asyncio.run(
        persistence.save_json(
            "file://mem/session:sess-1/summaries/session/2026-03-06T00-01-00Z.json",
            {
                "summary_doc_id": "summary-other",
                "scope_id": "session:sess-1",
                "summary_tag": "session",
                "summary": "other summary",
                "ts": "2026-03-06T00:01:00+00:00",
                "org_id": "other-org",
                "user_id": "other-user",
            },
        )
    )

    resp = client.get("/api/v1/memory/summaries", params={"scope_id": "session:sess-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert [summary["summary_id"] for summary in body["summaries"]] == ["summary-local"]
