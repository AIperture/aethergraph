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
