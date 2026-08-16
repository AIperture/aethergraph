from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
import pytest

from aethergraph.config.context import set_current_settings
from aethergraph.config.loader import load_settings
from aethergraph.core.runtime.runtime_services import install_services
from aethergraph.server.app_factory import create_app
from aethergraph.storage.contracts import StorageScope


def _utc_now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def app(tmp_path):
    cfg = load_settings()
    cfg.embed.enabled = False
    set_current_settings(cfg)
    application = create_app(workspace=str(tmp_path), cfg=cfg, log_level="warning")
    install_services(application.state.container)
    asyncio.run(application.state.container.start_storage())
    yield application
    asyncio.run(application.state.container.close_storage())


@pytest.fixture
def client(app):
    return TestClient(app)


async def _append(
    app,
    *,
    event_id: str,
    occurred_at: datetime,
    scope: StorageScope,
    kind: str,
    data: dict,
    text: str | None = None,
    stage: str | None = None,
    tags: tuple[str, ...] = (),
    topic: str | None = None,
    severity: int | None = None,
    signal: float | None = None,
) -> None:
    bucket = StorageScope(
        org_id=scope.org_id,
        user_id=scope.user_id,
        session_id=scope.session_id,
    )
    memory = app.state.container.memory_factory.for_public_execution(
        bucket,
        logical_scope_id=f"session:{scope.session_id}",
        provenance_scope=scope,
    )
    await memory.canonical.append_event(
        event_id=event_id,
        occurred_at=occurred_at,
        kind=kind,
        stage=stage,
        topic=topic,
        text=text,
        tags=tags,
        payload={"data": data},
        metrics={"latency_ms": 10.0} if event_id == "e3" else {},
        severity=severity,
        signal=signal,
    )


async def _seed_events(app) -> None:
    base = _utc_now() - timedelta(minutes=10)
    common = {"org_id": "local", "user_id": "local", "session_id": "sess-1"}
    await _append(
        app,
        event_id="e1",
        occurred_at=base + timedelta(minutes=1),
        scope=StorageScope(
            **common,
            run_id="run-1",
            agent_id="agent-a",
            graph_id="graph-1",
            node_id="node-1",
        ),
        kind="chat_user",
        stage="user",
        text="hello world from the user",
        tags=("chat", "user"),
        data={"text": "hello world from the user"},
    )
    await _append(
        app,
        event_id="e2",
        occurred_at=base + timedelta(minutes=2),
        scope=StorageScope(
            **common,
            run_id="run-2",
            agent_id="agent-a",
            graph_id="graph-1",
            node_id="node-2",
        ),
        kind="chat_assistant",
        stage="assistant",
        text="assistant reply with some extra detail for truncation checks",
        tags=("chat", "assistant"),
        data={"text": "assistant reply with some extra detail for truncation checks"},
        severity=3,
        signal=0.7,
    )
    await _append(
        app,
        event_id="e3",
        occurred_at=base + timedelta(minutes=3),
        scope=StorageScope(
            **common,
            run_id="run-3",
            agent_id="agent-b",
            graph_id="graph-2",
            node_id="node-3",
        ),
        kind="tool_result",
        stage="tool",
        tags=("tool", "result"),
        topic="search",
        data={"value": 42, "text": "tool output"},
    )
    await _append(
        app,
        event_id="e4",
        occurred_at=base + timedelta(minutes=4),
        scope=StorageScope(**common, run_id="run-4"),
        kind="checkpoint",
        stage="system",
        tags=("system",),
        data={"state": "saved"},
    )


def test_list_memory_events_recent_first_and_uses_provider_cursor(app, client):
    asyncio.run(_seed_events(app))
    response = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "limit": 2},
    )
    assert response.status_code == 200
    first = response.json()
    assert [event["event_id"] for event in first["events"]] == ["e4", "e3"]
    assert first["next_cursor"]
    assert first["events"][0]["snippet"] == '{"state": "saved"}'
    assert first["events"][1]["tool"] == "search"

    response = client.get(
        "/api/v1/memory/events",
        params={
            "session_id": "sess-1",
            "limit": 2,
            "cursor": first["next_cursor"],
        },
    )
    assert response.status_code == 200
    second = response.json()
    assert [event["event_id"] for event in second["events"]] == ["e2", "e1"]
    assert second["next_cursor"] is None


def test_list_memory_events_applies_canonical_scope_and_tag_filters(app, client):
    asyncio.run(_seed_events(app))
    response = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "agent_id": "agent-a"},
    )
    assert response.status_code == 200
    assert [event["event_id"] for event in response.json()["events"]] == ["e2", "e1"]

    response = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-1", "tags": "tool"},
    )
    assert response.status_code == 200
    assert [event["event_id"] for event in response.json()["events"]] == ["e3"]
    assert response.json()["events"][0]["snippet"] == "tool output"


def test_list_memory_events_maps_deprecated_selector_without_provider_alias(app, client):
    asyncio.run(_seed_events(app))
    response = client.get(
        "/api/v1/memory/events",
        params={"scope_id": "session:sess-1", "kinds": "chat_assistant"},
    )
    assert response.status_code == 200
    event = response.json()["events"][0]
    assert event["event_id"] == "e2"
    assert event["scope_id"] == "session:sess-1"
    assert event["run_id"] == "run-2"
    assert event["graph_id"] == "graph-1"
    assert event["node_id"] == "node-2"
    assert event["severity"] == 3
    assert event["signal"] == 0.7


def test_list_memory_events_isolates_request_identity(app, client):
    now = _utc_now()
    asyncio.run(
        _append(
            app,
            event_id="tenant-a",
            occurred_at=now,
            scope=StorageScope(
                org_id="org-a",
                user_id="user-a",
                session_id="sess-tenant",
                run_id="run-a",
            ),
            kind="user_msg",
            text="from tenant a",
            data={},
        )
    )
    asyncio.run(
        _append(
            app,
            event_id="tenant-b",
            occurred_at=now,
            scope=StorageScope(
                org_id="org-b",
                user_id="user-b",
                session_id="sess-tenant",
                run_id="run-b",
            ),
            kind="user_msg",
            text="from tenant b",
            data={},
        )
    )
    response = client.get(
        "/api/v1/memory/events",
        params={"session_id": "sess-tenant"},
        headers={"X-User-ID": "user-a", "X-Org-ID": "org-a", "X-Mode": "cloud"},
    )
    assert response.status_code == 200
    assert [event["event_id"] for event in response.json()["events"]] == ["tenant-a"]


def test_memory_summaries_and_search_use_canonical_events(app, client):
    memory = app.state.container.memory_factory.for_public_execution(
        StorageScope(org_id="local", user_id="local", session_id="sess-1"),
        logical_scope_id="session:sess-1",
        provenance_scope=StorageScope(
            session_id="sess-1",
            run_id="run-1",
            org_id="local",
            user_id="local",
        ),
    )
    asyncio.run(
        memory.append_event(
            kind="long_term_summary",
            data={"summary": "canonical migration summary"},
            text="canonical migration summary",
            tags=["summary", "session"],
        )
    )
    response = client.get(
        "/api/v1/memory/summaries",
        params={"scope_id": "session:sess-1"},
    )
    assert response.status_code == 200
    assert response.json()["summaries"][0]["text"] == "canonical migration summary"
    assert response.json()["next_cursor"] is None

    response = client.post(
        "/api/v1/memory/search",
        json={"scope_id": "session:sess-1", "query": "migration", "top_k": 5},
    )
    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["event"] is None
    assert hit["summary"]["text"] == "canonical migration summary"
