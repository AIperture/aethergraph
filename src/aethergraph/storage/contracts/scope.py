"""Canonical provider-neutral storage scope."""

from __future__ import annotations

from dataclasses import dataclass, fields

from .errors import StorageScopeError


@dataclass(frozen=True, slots=True)
class StorageScope:
    """Immutable normalized identity dimensions accepted by storage providers."""

    tenant_id: str | None = None
    project_id: str | None = None
    org_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    scope_key: str | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise StorageScopeError(f"{item.name} must be a non-empty string when supplied")

    def require(self, *names: str) -> StorageScope:
        """Require exact canonical dimensions for one storage operation.

        The scope is immutable. This validation returns the same instance when every
        named dimension is present and fails closed otherwise.

        Examples:
            Require tenant and project ownership:
                ```python
                scope.require("tenant_id", "project_id")
                ```

            Require a run dimension:
                ```python
                run_scope = StorageScope(run_id="run-1")
                run_scope.require("run_id")
                ```

        Args:
            names: Exact `StorageScope` field names required by the caller.

        Returns:
            StorageScope: The unchanged validated scope instance.

        Notes:
            Unknown field names and absent values raise `StorageScopeError`.
        """
        valid = {item.name for item in fields(self)}
        unknown = tuple(name for name in names if name not in valid)
        if unknown:
            raise StorageScopeError(f"Unknown storage scope dimensions: {', '.join(unknown)}")
        missing = tuple(name for name in names if getattr(self, name) is None)
        if missing:
            raise StorageScopeError(f"Missing required storage scope: {', '.join(missing)}")
        return self

    def as_filter(self) -> dict[str, str]:
        """Return the populated canonical dimensions as a new mapping.

        The returned mapping is detached from the immutable scope and omits every
        dimension whose value is `None`.

        Examples:
            Build a tenant filter:
                ```python
                StorageScope(tenant_id="tenant-1").as_filter()
                ```

            Build an empty filter:
                ```python
                StorageScope().as_filter()
                ```

        Args:
            None.

        Returns:
            dict[str, str]: Populated canonical field names and values.

        Notes:
            `app_id`, `client_id`, and legacy memory scope aliases are intentionally
            absent from this contract.
        """
        return {
            item.name: value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }


def storage_scope_matches_filter(
    record_scope: StorageScope,
    filter_scope: StorageScope,
) -> bool:
    """Return whether one persisted scope matches a populated caller filter.

    The filter may contain fewer dimensions than the persisted record. Every
    populated filter dimension must match exactly, and an empty filter fails closed.

    Examples:
        Match a project filter against a session record:
            ```python
            record = StorageScope(project_id="project-1", session_id="session-1")
            assert storage_scope_matches_filter(
                record,
                StorageScope(project_id="project-1"),
            )
            ```

        Reject an empty filter:
            ```python
            assert not storage_scope_matches_filter(
                StorageScope(project_id="project-1"),
                StorageScope(),
            )
            ```

    Args:
        record_scope: Complete persisted scope being authorized.
        filter_scope: Caller-supplied populated scope constraints.

    Returns:
        bool: `True` only when every populated filter dimension matches the record.

    Notes:
        This relation is for record lookup and mutation authorization. Use
        `storage_scope_covers` for parent-to-child execution scope validation.
    """
    filters = filter_scope.as_filter()
    return bool(filters) and all(
        getattr(record_scope, name) == value for name, value in filters.items()
    )


def storage_scope_covers(
    parent_scope: StorageScope,
    child_scope: StorageScope,
) -> bool:
    """Return whether a populated parent scope covers a narrower child scope.

    The child may add execution provenance such as run, graph, or node dimensions.
    Every populated parent dimension must remain present and equal in the child.

    Examples:
        Cover node provenance from a session scope:
            ```python
            parent = StorageScope(project_id="project-1", session_id="session-1")
            child = StorageScope(
                project_id="project-1",
                session_id="session-1",
                graph_id="graph-1",
                node_id="node-1",
            )
            assert storage_scope_covers(parent, child)
            ```

        Reject a child from another session:
            ```python
            assert not storage_scope_covers(
                StorageScope(project_id="project-1", session_id="session-1"),
                StorageScope(project_id="project-1", session_id="session-2"),
            )
            ```

    Args:
        parent_scope: Populated owner, run, or session scope establishing authority.
        child_scope: Narrower operation or occurrence scope carrying provenance.

    Returns:
        bool: `True` only when every populated parent dimension matches the child.

    Notes:
        Empty parent scopes fail closed. This relation does not authorize record
        filters; use `storage_scope_matches_filter` for that operation.
    """
    parent = parent_scope.as_filter()
    return bool(parent) and all(
        getattr(child_scope, name) == value for name, value in parent.items()
    )
