"""Canonical scope construction and validation for graph-state persistence."""

from __future__ import annotations

from typing import Any

from aethergraph.storage.contracts.errors import StorageScopeError
from aethergraph.storage.contracts.scope import StorageScope


def scope_for_runtime_env(env: Any, *, run_id: str | None = None) -> StorageScope:
    """Build canonical graph-state scope from one runtime environment.

    Intro:
        Copies stable graph ownership and identity dimensions from the runtime boundary
        and optionally substitutes the run dimension for an explicit resume source.

    Examples:
        Scope the active run:
        ```python
        scope = scope_for_runtime_env(env)
        ```

        Scope a resume source:
        ```python
        source_scope = scope_for_runtime_env(env, run_id="source-run")
        ```

    Args:
        env: Runtime environment carrying identity and graph dimensions.
        run_id: Optional exact run identity replacing the active run identity.

    Returns:
        StorageScope: Immutable canonical graph-state scope.

    Notes:
        Session, Agent, deprecated `app_id`, and request `client_id` are provenance rather
        than graph-state identity and are intentionally omitted.
    """
    identity = getattr(env, "identity", None)
    return StorageScope(
        org_id=getattr(identity, "org_id", None),
        user_id=getattr(identity, "user_id", None),
        run_id=run_id or getattr(env, "run_id", None),
        graph_id=getattr(env, "graph_id", None),
    ).require("run_id")


def scope_for_run_record(record: Any) -> StorageScope:
    """Build canonical graph-state scope from one durable run record.

    Intro:
        Projects stable graph ownership and identity dimensions into the shared storage
        scope contract without adding compatibility aliases or provenance dimensions.

    Examples:
        Scope a persisted run:
        ```python
        scope = scope_for_run_record(record)
        ```

        Read its graph snapshot:
        ```python
        snapshot = await store.load_latest_snapshot(scope_for_run_record(record), record.run_id)
        ```

    Args:
        record: Durable run record containing canonical owner and execution fields.

    Returns:
        StorageScope: Immutable canonical graph-state scope.

    Notes:
        Session, Agent, and deprecated `app_id` remain run provenance and do not alter
        graph-state identity.
    """
    return StorageScope(
        org_id=getattr(record, "org_id", None),
        user_id=getattr(record, "user_id", None),
        run_id=getattr(record, "run_id", None),
        graph_id=getattr(record, "graph_id", None),
    ).require("run_id")


def require_graph_run_scope(
    scope: StorageScope,
    *,
    run_id: str,
    graph_id: str | None = None,
) -> StorageScope:
    """Validate an exact canonical scope against graph-state domain identity.

    Intro:
        Fails closed when the required run dimension or an explicitly supplied graph
        dimension disagrees with the graph snapshot or event being accessed.

    Examples:
        Validate one snapshot:
        ```python
        require_graph_run_scope(scope, run_id=snapshot.run_id, graph_id=snapshot.graph_id)
        ```

        Validate one run read:
        ```python
        require_graph_run_scope(scope, run_id="run-1")
        ```

    Args:
        scope: Canonical scope supplied at the service boundary.
        run_id: Exact graph-state run identity.
        graph_id: Optional exact graph identity.

    Returns:
        StorageScope: The unchanged validated scope.

    Notes:
        Missing or mismatched dimensions raise `StorageScopeError`; no lookup is attempted.
    """
    scope.require("run_id")
    if scope.run_id != run_id:
        raise StorageScopeError(
            f"Graph-state run scope mismatch: scope={scope.run_id!r}, value={run_id!r}"
        )
    if graph_id is not None and scope.graph_id is not None and scope.graph_id != graph_id:
        raise StorageScopeError(
            f"Graph-state graph scope mismatch: scope={scope.graph_id!r}, value={graph_id!r}"
        )
    return scope
