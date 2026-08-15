"""Shared trusted-scope rules for inactive canonical service projections."""

from __future__ import annotations

from aethergraph.storage.contracts import StorageScope

_EXECUTION_OR_EXTERNAL_DIMENSIONS = ("session_id", "run_id", "node_id", "scope_key")


def validate_storage_owner_scope(scope: StorageScope) -> None:
    if not scope.as_filter():
        raise ValueError("owner_scope must contain at least one canonical dimension")
    populated = tuple(
        name for name in _EXECUTION_OR_EXTERNAL_DIMENSIONS if getattr(scope, name) is not None
    )
    if populated:
        raise ValueError(
            "owner_scope contains execution/external dimensions: " + ", ".join(populated)
        )


def merge_storage_scope(owner_scope: StorageScope, **dimensions: str) -> StorageScope:
    validate_storage_owner_scope(owner_scope)
    values = owner_scope.as_filter()
    for name, value in dimensions.items():
        if name in values and values[name] != value:
            raise ValueError(f"service scope conflicts with owner_scope {name}")
        values[name] = value
    return StorageScope(**values)
