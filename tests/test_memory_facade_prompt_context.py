from __future__ import annotations

from dataclasses import replace

import pytest

from aethergraph.contracts.services.memory import Event
from aethergraph.services.memory.facade.core import MemoryFacade
from aethergraph.services.scope.scope import Scope


class FakeHotLog:
    def __init__(self, events: list[Event] | None = None) -> None:
        self.events = list(events or [])
        self.appended: list[Event] = []

    async def append(self, timeline_id: str, evt: Event, *, ttl_s: int, limit: int) -> None:
        self.events.append(evt)
        self.appended.append(evt)

    async def query(
        self,
        timeline_id: str,
        *,
        tenant=None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        since=None,
        until=None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Event]:
        rows = list(self.events)
        if kinds is not None:
            rows = [e for e in rows if e.kind in kinds]
        if tags:
            want = set(tags)
            rows = [e for e in rows if want.issubset(set(e.tags or []))]
        if session_id is not None:
            rows = [e for e in rows if e.session_id == session_id]
        if run_id is not None:
            rows = [e for e in rows if e.run_id == run_id]
        if agent_id is not None:
            rows = [e for e in rows if e.agent_id == agent_id]
        rows = rows[offset:]
        return rows[:limit]


class FakePersistence:
    def __init__(self, events: list[Event] | None = None) -> None:
        self.events = list(events or [])

    async def append_event(self, timeline_id: str, evt: Event) -> None:
        self.events.append(evt)

    async def save_json(self, uri: str, obj: dict) -> str:
        return uri

    async def load_json(self, uri: str) -> dict:
        raise NotImplementedError

    async def get_events_by_ids(self, timeline_id: str, event_ids: list[str], tenant=None) -> list[Event]:
        wanted = set(event_ids)
        return [e for e in self.events if e.event_id in wanted]

    async def query_events(
        self,
        timeline_id: str,
        *,
        tenant=None,
        since: str | None = None,
        until: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Event]:
        rows = list(self.events)
        if kinds is not None:
            rows = [e for e in rows if e.kind in kinds]
        if tags:
            want = set(tags)
            rows = [e for e in rows if want.issubset(set(e.tags or []))]
        if session_id is not None:
            rows = [e for e in rows if e.session_id == session_id]
        if run_id is not None:
            rows = [e for e in rows if e.run_id == run_id]
        if agent_id is not None:
            rows = [e for e in rows if e.agent_id == agent_id]
        rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows


class FakeArtifactStore:
    pass


def _evt(
    event_id: str,
    *,
    ts: str,
    kind: str,
    scope_id: str,
    session_id: str | None,
    run_id: str,
    text: str | None = None,
    stage: str | None = None,
    tags: list[str] | None = None,
    data: dict | None = None,
    tool: str | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        ts=ts,
        run_id=run_id,
        scope_id=scope_id,
        session_id=session_id,
        user_id="user-1",
        org_id="org-1",
        kind=kind,
        stage=stage,
        text=text,
        tags=tags or [],
        data=data,
        tool=tool,
    )


def _make_facade(*, hotlog_events: list[Event], persisted_events: list[Event]) -> MemoryFacade:
    scope = Scope(
        org_id="org-1",
        user_id="user-1",
        session_id="sess-1",
        run_id="run-current",
        memory_level="user",
    )
    return MemoryFacade(
        run_id="run-current",
        session_id="sess-1",
        graph_id=None,
        node_id=None,
        scope=scope,
        hotlog=FakeHotLog(hotlog_events),
        persistence=FakePersistence(persisted_events),
        scoped_indices=None,
        artifact_store=FakeArtifactStore(),
    )


@pytest.mark.asyncio
async def test_chat_history_for_llm_uses_configured_retrieval_source_and_summary_scope() -> None:
    facade = _make_facade(
        hotlog_events=[
            _evt(
                "chat-hot",
                ts="2026-03-07T10:00:00Z",
                kind="chat.turn",
                scope_id="session:sess-1",
                session_id="sess-1",
                run_id="run-hot",
                text="from hotlog",
                stage="user",
                tags=["chat"],
                data={"text": "from hotlog", "role": "user"},
            )
        ],
        persisted_events=[
            _evt(
                "summary-s1",
                ts="2026-03-07T09:00:00Z",
                kind="long_term_summary",
                scope_id="session:sess-1",
                session_id="sess-1",
                run_id="run-summary-1",
                text="summary for session one",
                stage="summary",
                tags=["summary", "session"],
                data={"summary": "summary for session one", "scope_id": "session:sess-1"},
            ),
            _evt(
                "summary-s2",
                ts="2026-03-07T09:05:00Z",
                kind="long_term_summary",
                scope_id="session:sess-2",
                session_id="sess-2",
                run_id="run-summary-2",
                text="summary for session two",
                stage="summary",
                tags=["summary", "session"],
                data={"summary": "summary for session two", "scope_id": "session:sess-2"},
            ),
            _evt(
                "chat-persisted",
                ts="2026-03-07T09:30:00Z",
                kind="chat.turn",
                scope_id="session:sess-1",
                session_id="sess-1",
                run_id="run-persisted",
                text="from persistence",
                stage="assistant",
                tags=["chat"],
                data={"text": "from persistence", "role": "assistant"},
            ),
        ],
    )

    hotlog_history = await facade.chat_history_for_llm(
        summary_scope_id="session:sess-1",
        level="scope",
        use_persistence=False,
    )
    persisted_history = await facade.chat_history_for_llm(
        summary_scope_id="session:sess-1",
        level="scope",
        use_persistence=True,
    )

    assert hotlog_history["summary"] == "summary for session one"
    assert hotlog_history["messages"][-1] == {"role": "user", "content": "from hotlog"}
    assert all(msg["content"] != "from persistence" for msg in hotlog_history["messages"])

    assert persisted_history["summary"] == "summary for session one"
    assert persisted_history["messages"][-1] == {"role": "assistant", "content": "from persistence"}
    assert all(msg["content"] != "from hotlog" for msg in persisted_history["messages"])


@pytest.mark.asyncio
async def test_build_prompt_segments_applies_level_filter_to_recent_tools() -> None:
    facade = _make_facade(
        hotlog_events=[
            _evt(
                "chat-s1",
                ts="2026-03-07T10:00:00Z",
                kind="chat.turn",
                scope_id="session:sess-1",
                session_id="sess-1",
                run_id="run-current",
                text="session one chat",
                stage="user",
                tags=["chat"],
                data={"text": "session one chat", "role": "user"},
            ),
            _evt(
                "tool-s1",
                ts="2026-03-07T10:01:00Z",
                kind="tool_result",
                scope_id="session:sess-1",
                session_id="sess-1",
                run_id="run-current",
                text="tool in current session",
                tags=["tool", "session"],
                tool="search",
            ),
            _evt(
                "tool-s2",
                ts="2026-03-07T10:02:00Z",
                kind="tool_result",
                scope_id="session:sess-2",
                session_id="sess-2",
                run_id="run-other",
                text="tool in other session",
                tags=["tool", "session"],
                tool="search",
            ),
        ],
        persisted_events=[],
    )

    segments = await facade.build_prompt_segments(
        include_long_term=False,
        include_recent_tools=True,
        tool="search",
        recent_chat_limit=5,
        tool_limit=5,
        level="session",
        use_persistence=False,
    )

    assert [item["text"] for item in segments["recent_chat"]] == ["session one chat"]
    assert [item["message"] for item in segments["recent_tools"]] == ["tool in current session"]


@pytest.mark.asyncio
async def test_soft_hydrate_last_summary_honors_scope_id() -> None:
    summary_one = _evt(
        "summary-s1",
        ts="2026-03-07T09:00:00Z",
        kind="long_term_summary",
        scope_id="session:sess-1",
        session_id="sess-1",
        run_id="run-summary-1",
        text="summary for session one",
        stage="summary",
        tags=["summary", "session"],
        data={"summary": "summary for session one", "scope_id": "session:sess-1"},
    )
    summary_two = replace(
        summary_one,
        event_id="summary-s2",
        ts="2026-03-07T09:05:00Z",
        scope_id="session:sess-2",
        session_id="sess-2",
        run_id="run-summary-2",
        text="summary for session two",
        data={"summary": "summary for session two", "scope_id": "session:sess-2"},
    )
    facade = _make_facade(hotlog_events=[], persisted_events=[summary_one, summary_two])

    hydrated = await facade.soft_hydrate_last_summary(
        summary_tag="session",
        scope_id="session:sess-2",
        level="scope",
    )

    assert hydrated == {"summary": "summary for session two", "scope_id": "session:sess-2"}
    assert facade.hotlog.appended[-1].kind == "long_term_summary_hydrate"
    assert facade.hotlog.appended[-1].data == {"summary": hydrated}
