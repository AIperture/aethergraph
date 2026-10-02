"""Provider-neutral identity and failures for canonical inspection."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from aethergraph.storage.contracts import StorageError, StorageMigrationRequiredError


class ObservabilityUnavailableError(RuntimeError):
    """Signal that the canonical observation service is unavailable."""


class ObservabilityNotFoundError(LookupError):
    """Signal that a scoped canonical inspection record does not exist."""


class ObservabilityIndexRequiredError(ObservabilityUnavailableError):
    """Signal that compact inspection requires writable index preparation."""


@contextmanager
def observability_storage_errors() -> Iterator[None]:
    """Translate provider failures across a composed public inspection operation.

    Examples:
        ```python
        with observability_storage_errors():
            page = await facade.page_observation_records(run_id="run-1", categories=("log",))
        ```
        ```python
        with observability_storage_errors():
            page = await engine_projection.summary()
        ```

    Args:
        None.

    Returns:
        Iterator[None]: A context boundary that preserves successful results.

    Notes:
        Covers supporting stores used by higher-level inspection owners. Opening
        workspace migrations retain their separate public workspace error. Provider
        paths and private failure details remain in the exception cause only.
    """
    try:
        yield
    except StorageMigrationRequiredError as exc:
        raise ObservabilityIndexRequiredError(
            "Compact inspection requires writable index preparation."
        ) from exc
    except StorageError as exc:
        raise ObservabilityUnavailableError("Canonical inspection storage is unavailable.") from exc


class ObservabilityWorkspaceError(RuntimeError):
    """Signal that a manifested historical workspace cannot be opened."""


class ObservabilityMigrationRequiredError(ObservabilityWorkspaceError):
    """Signal that intact historical storage needs a writable migration before inspection."""


@dataclass(frozen=True)
class ObservabilityIdentity:
    """Authenticated identity applied to one canonical inspection reader."""

    mode: str = "local"
    user_id: str | None = None
    org_id: str | None = None


__all__ = [
    "ObservabilityIdentity",
    "ObservabilityIndexRequiredError",
    "ObservabilityMigrationRequiredError",
    "ObservabilityNotFoundError",
    "ObservabilityUnavailableError",
    "ObservabilityWorkspaceError",
    "observability_storage_errors",
]
