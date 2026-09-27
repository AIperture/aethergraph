"""Exact question status remains observable after its parent finishes."""

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_embedded_runtime import _container
from test_resume_router import store  # noqa: F401

from aethergraph.runtime import EmbeddedRuntime, RuntimeInteractionError
from aethergraph.services.continuations.continuation import (
    ContinuationDraft,
    ContinuationStatus,
    Correlator,
)
from aethergraph.services.integration.interactions import (
    InteractionResolutionError,
    InteractionResolver,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["waiting", "resumed", "expired", "canceled", "overdue", "timer"]
)
async def test_exact_inspection_preserves_canonical_state_and_authority(store, outcome):  # noqa: F811
    now = datetime.now(UTC)
    created = await store.create(
        ContinuationDraft(
            run_id="child-run",
            node_id="node",
            session_id="child-session",
            kind="user_input",
            created_at=now - timedelta(seconds=2),
            deadline=now + timedelta(seconds=-1 if outcome == "overdue" else 60),
            correlators=(Correlator(scheme="interaction", channel="public", message="question"),),
        )
    )
    wait = created.record
    if outcome not in {"waiting", "overdue"}:
        wait = await store.update(
            replace(
                wait,
                revision=2,
                closed_at=now,
                status=ContinuationStatus.RESUMED
                if outcome == "timer"
                else ContinuationStatus(outcome),
                payload={"timer_kind": "deadline"}
                if outcome == "timer"
                else {"text": "private answer"},
            ),
            expected_revision=1,
        )
    manager = SimpleNamespace(
        get_record=AsyncMock(
            return_value=SimpleNamespace(
                run_id="child-run",
                session_id="child-session",
                parent=SimpleNamespace(run_id="completed-parent", session_id="parent-session"),
            )
        )
    )
    runtime = EmbeddedRuntime(_container(cont_store=store, run_manager=manager))
    observed = await runtime.inspect_interaction(
        session_id="parent-session", interaction_id="question"
    )
    assert observed.status == ("expired" if outcome in {"overdue", "timer"} else outcome)
    assert observed.revision == wait.revision
    assert set(asdict(observed)) == {
        "interaction_id",
        "status",
        "revision",
        "deadline",
        "closed_at",
    }
    assert await store.get_by_id("child-run", "node", wait.continuation_id) == wait
    resolver = InteractionResolver(store, run_manager=manager)
    if outcome == "waiting":
        assert (
            await resolver.resolve_exact(
                session_id="parent-session",
                interaction_id="question",
                expected_kinds={"user_input"},
            )
        ).continuation == wait
    else:
        with pytest.raises(InteractionResolutionError) as error:
            await resolver.resolve_exact(
                session_id="parent-session",
                interaction_id="question",
                expected_kinds={"user_input"},
            )
        assert error.value.code == "integration.interaction_not_found"
    with pytest.raises(RuntimeInteractionError) as error:
        await runtime.inspect_interaction(session_id="unrelated", interaction_id="question")
    assert error.value.code == "integration.interaction_session_mismatch"
    with pytest.raises(RuntimeInteractionError) as error:
        await runtime.inspect_interaction(session_id="parent-session", interaction_id="missing")
    assert error.value.code == "integration.interaction_not_found"
