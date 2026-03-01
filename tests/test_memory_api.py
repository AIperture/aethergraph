# tests/test_memory_api.py

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
    # Load config and build app like other API tests
    cfg = load_settings()
    set_current_settings(cfg)
    app = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")

    # Ensure container services are globally installed for current process
    install_services(app.state.container)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.mark.asyncio
async def _seed_events_for_scope(app, scope_id: str):
    """Helper to seed some events into HotLog via the container's memory_factory."""
    container = app.state.container
    mem_factory = container.memory_factory
    hotlog = mem_factory.hotlog

    # Three events with different kinds/tags/timestamps
    base_ts = _utc_now() - timedelta(minutes=10)

    evts = [
        Event(
            event_id="e1",
            ts=(base_ts + timedelta(minutes=1)).isoformat(),
            run_id=scope_id,
            scope_id=scope_id,
            kind="chat_user",
            stage="user",
            text="hello world",
            tags=["chat", "user"],
            data={"text": "hello world"},
            metrics=None,
        ),
        Event(
            event_id="e2",
            ts=(base_ts + timedelta(minutes=2)).isoformat(),
            run_id=scope_id,
            scope_id=scope_id,
            kind="chat_assistant",
            stage="assistant",
            text="assistant reply",
            tags=["chat", "assistant"],
            data={"text": "assistant reply"},
            metrics=None,
        ),
        Event(
            event_id="e3",
            ts=(base_ts + timedelta(minutes=3)).isoformat(),
            run_id=scope_id,
            scope_id=scope_id,
            kind="tool_result",
            stage="tool",
            text="tool output",
            tags=["tool", "result"],
            data={"value": 42},
            metrics={"latency_ms": 10.0},
        ),
    ]

    for evt in evts:
        # use a generous TTL and the factory's hot_limit
        await hotlog.append(scope_id, evt, ttl_s=3600, limit=mem_factory.hot_limit)


def test_list_memory_events_basic(app, client):
    scope_id = "scope-memory-1"

    # Seed events in async world first
    asyncio.run(_seed_events_for_scope(app, scope_id))

    resp = client.get(
        "/api/v1/memory/events",
        params={"scope_id": scope_id},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "events" in data
    events = data["events"]
    assert len(events) == 3

    # Check structure of first event
    e0 = events[0]
    assert e0["scope_id"] == scope_id
    assert "event_id" in e0
    assert "created_at" in e0
    assert isinstance(e0["tags"], list)


def test_list_memory_events_with_filters(app, client):
    scope_id = "scope-memory-2"
    asyncio.run(_seed_events_for_scope(app, scope_id))

    # Filter by kind
    resp = client.get(
        "/api/v1/memory/events",
        params={"scope_id": scope_id, "kinds": "chat_user"},
    )
    assert resp.status_code == 200
    data = resp.json()
    events = data["events"]
    assert len(events) == 1
    assert events[0]["kind"] == "chat_user"

    # Filter by tags (require "tool")
    resp2 = client.get(
        "/api/v1/memory/events",
        params={"scope_id": scope_id, "tags": "tool"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    events2 = data2["events"]
    assert len(events2) == 1
    assert "tool" in events2[0]["tags"]

    # Time filter: after a very late time -> zero results
    far_future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    resp3 = client.get(
        "/api/v1/memory/events",
        params={"scope_id": scope_id, "after": far_future},
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["events"] == []
