from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import session as session_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import SemanticEventKind
from aethergraph.core.runtime.run_types import SessionKind
from tests._canonical_storage_fakes import make_session_store


class FakeSemanticEventStore:
    def __init__(self, rows: list[Any] | None = None):
        self.rows = list(rows or [])

    async def list_session(self, *, deployment_id: str, session_id: str):
        assert deployment_id == "deployment-1"
        return [row for row in self.rows if row.event.session_id == session_id]


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: list[dict[str, Any]], **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.response, {"prompt_tokens": 12, "completion_tokens": 6}


class FakeLLMService:
    def __init__(self, response: str):
        self.client = FakeLLMClient(response)

    def get(self, name: str = "default") -> FakeLLMClient:
        assert name == "default"
        return self.client


class FakeContainer:
    def __init__(self, *, session_store, semantic_events, llm=None):
        self.session_store = session_store
        self.semantic_events = semantic_events
        self.host_manifest = SimpleNamespace(deployment_id="deployment-1")
        self.llm = llm


def _identity(
    *,
    user_id: str = "user-1",
    org_id: str = "org-1",
    mode: str = "local",
) -> RequestIdentity:
    return RequestIdentity(user_id=user_id, org_id=org_id, mode=mode)


def _build_app(
    container: FakeContainer,
    *,
    identity: RequestIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    app = FastAPI()
    app.include_router(session_api.router, prefix="/api/v1")

    async def fake_identity() -> RequestIdentity:
        return identity

    app.dependency_overrides[session_api.get_identity] = fake_identity
    monkeypatch.setattr(session_api, "current_services", lambda: container)
    return app


def _chat_row(
    *,
    session_id: str,
    row_id: int,
    event_id: str,
    ts: float,
    event_type: str,
    text: str | None,
):
    del row_id, event_id, ts
    kind = (
        SemanticEventKind.INPUT_ACCEPTED
        if event_type == "user.message"
        else SemanticEventKind.MESSAGE_COMPLETED
    )
    return SimpleNamespace(
        event=SimpleNamespace(
            session_id=session_id,
            kind=kind,
            payload=SimpleNamespace(text=text),
        )
    )


def test_infer_session_title_generates_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_session_store()
    created = __import__("asyncio").run(
        store.create(
            kind=SessionKind.chat,
            title=None,
            external_ref=None,
            user_id="user-1",
            org_id="org-1",
            source="test",
        )
    )
    rows = [
        _chat_row(
            session_id=created.session_id,
            row_id=1,
            event_id="evt-1",
            ts=1.0,
            event_type="user.message",
            text="Help me build a Slack support bot for onboarding.",
        ),
        _chat_row(
            session_id=created.session_id,
            row_id=2,
            event_id="evt-2",
            ts=2.0,
            event_type="agent.message",
            text="I can outline the architecture and rollout plan.",
        ),
    ]
    container = FakeContainer(
        session_store=store,
        semantic_events=FakeSemanticEventStore(rows),
        llm=FakeLLMService('"Slack Onboarding Bot Plan"'),
    )
    app = _build_app(container, identity=_identity(), monkeypatch=monkeypatch)
    client = TestClient(app)

    resp = client.post(f"/api/v1/sessions/{created.session_id}/infer-title", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "session_id": created.session_id,
        "title": "Slack Onboarding Bot Plan",
        "updated": True,
        "reason": "generated",
    }

    stored = __import__("asyncio").run(store.get(created.session_id))
    assert stored is not None
    assert stored.title == "Slack Onboarding Bot Plan"
    assert stored.title_source == "auto"
    assert len(container.llm.client.calls) == 1


def test_infer_session_title_skips_when_not_enough_context(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_session_store()
    created = __import__("asyncio").run(
        store.create(
            kind=SessionKind.chat,
            title=None,
            external_ref=None,
            user_id="user-1",
            org_id="org-1",
            source="test",
        )
    )
    rows = [
        _chat_row(
            session_id=created.session_id,
            row_id=1,
            event_id="evt-1",
            ts=1.0,
            event_type="user.message",
            text="Summarize this file.",
        )
    ]
    container = FakeContainer(
        session_store=store,
        semantic_events=FakeSemanticEventStore(rows),
        llm=FakeLLMService("unused"),
    )
    app = _build_app(container, identity=_identity(), monkeypatch=monkeypatch)
    client = TestClient(app)

    resp = client.post(f"/api/v1/sessions/{created.session_id}/infer-title", json={})
    assert resp.status_code == 200
    assert resp.json()["reason"] == "skipped_no_context"
    assert container.llm.client.calls == []


def test_infer_session_title_normalizes_output(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_session_store()
    created = __import__("asyncio").run(
        store.create(
            kind=SessionKind.chat,
            title=None,
            external_ref=None,
            user_id="user-1",
            org_id="org-1",
            source="test",
        )
    )
    rows = [
        _chat_row(
            session_id=created.session_id,
            row_id=1,
            event_id="evt-1",
            ts=1.0,
            event_type="user.message",
            text="Help me analyze solar inverter telemetry anomalies.",
        ),
        _chat_row(
            session_id=created.session_id,
            row_id=2,
            event_id="evt-2",
            ts=2.0,
            event_type="agent.message",
            text="I'll inspect the symptoms and propose likely causes.",
        ),
    ]
    container = FakeContainer(
        session_store=store,
        semantic_events=FakeSemanticEventStore(rows),
        llm=FakeLLMService('Title:\n"Solar Inverter Anomaly Review"'),
    )
    app = _build_app(container, identity=_identity(), monkeypatch=monkeypatch)
    client = TestClient(app)

    resp = client.post(f"/api/v1/sessions/{created.session_id}/infer-title", json={})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Solar Inverter Anomaly Review"


def test_manual_rename_marks_title_source_and_allows_explicit_ai_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_session_store()
    created = __import__("asyncio").run(
        store.create(
            kind=SessionKind.chat,
            title=None,
            external_ref=None,
            user_id="user-1",
            org_id="org-1",
            source="test",
        )
    )
    rows = [
        _chat_row(
            session_id=created.session_id,
            row_id=1,
            event_id="evt-1",
            ts=1.0,
            event_type="user.message",
            text="Draft a launch email.",
        ),
        _chat_row(
            session_id=created.session_id,
            row_id=2,
            event_id="evt-2",
            ts=2.0,
            event_type="agent.message",
            text="I'll propose a crisp customer-facing draft.",
        ),
    ]
    container = FakeContainer(
        session_store=store,
        semantic_events=FakeSemanticEventStore(rows),
        llm=FakeLLMService("Launch Email Draft"),
    )
    app = _build_app(container, identity=_identity(), monkeypatch=monkeypatch)
    client = TestClient(app)

    rename_resp = client.patch(
        f"/api/v1/sessions/{created.session_id}",
        json={"title": "My Manual Name"},
    )
    assert rename_resp.status_code == 200
    assert rename_resp.json()["title_source"] == "manual"

    infer_resp = client.post(
        f"/api/v1/sessions/{created.session_id}/infer-title",
        json={"force": True, "mode": "refresh"},
    )
    assert infer_resp.status_code == 200
    body = infer_resp.json()
    assert body["updated"] is True
    assert body["reason"] == "generated"
    assert body["title"] == "Launch Email Draft"
    assert len(container.llm.client.calls) == 1


def test_infer_session_title_enforces_session_access(monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_session_store()
    created = __import__("asyncio").run(
        store.create(
            kind=SessionKind.chat,
            title=None,
            external_ref=None,
            user_id="owner-1",
            org_id="org-1",
            source="test",
        )
    )
    rows = [
        _chat_row(
            session_id=created.session_id,
            row_id=1,
            event_id="evt-1",
            ts=1.0,
            event_type="user.message",
            text="Help me plan a migration.",
        ),
        _chat_row(
            session_id=created.session_id,
            row_id=2,
            event_id="evt-2",
            ts=2.0,
            event_type="agent.message",
            text="I can break this into safe rollout steps.",
        ),
    ]
    container = FakeContainer(
        session_store=store,
        semantic_events=FakeSemanticEventStore(rows),
        llm=FakeLLMService("Migration Rollout Plan"),
    )
    app = _build_app(
        container,
        identity=_identity(user_id="other-user", org_id="org-1", mode="cloud"),
        monkeypatch=monkeypatch,
    )
    client = TestClient(app)

    resp = client.post(f"/api/v1/sessions/{created.session_id}/infer-title", json={})
    assert resp.status_code == 403
