"""Continuation construction invariants."""

from datetime import UTC

from aethergraph.services.continuations.continuation import Continuation


def _continuation(token: str) -> Continuation:
    return Continuation(
        run_id="run-1",
        node_id="node-1",
        kind="text",
        token=token,
    )


def test_continuations_receive_independent_timezone_aware_creation_times() -> None:
    first = _continuation("first")
    second = _continuation("second")

    assert first.created_at is not second.created_at
    assert first.created_at.tzinfo is UTC
    assert second.created_at.tzinfo is UTC
