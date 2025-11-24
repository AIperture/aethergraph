from typing import Any, Literal, Protocol

from aethergraph.contracts.services.artifacts import Artifact

"""
Artifact index interface for storing and retrieving artifact metadata.
This is a special index used for tracking artifacts generated during runs.

Typical implementations include:
- FileSystemArtifactIndex: File system-based artifact index for durable storage
- DatabaseArtifactIndex: (future) Database-backed artifact index for scalable storage and querying

Note Artifact index is a specialized index for artifacts, different from general document or blob stores. 
"""


class AsyncArtifactIndex(Protocol):
    """Backend-agnostic index for artifact metadata & occurrences."""

    async def upsert(self, a: Artifact) -> None:
        """Insert or update a single artifact record."""
        ...

    async def list_for_run(self, run_id: str) -> list[Artifact]:
        """Return all artifacts for a given run_id."""
        ...

    async def search(
        self,
        *,
        kind: str | None = None,
        labels: dict[str, Any] | None = None,
        metric: str | None = None,
        mode: Literal["max", "min"] | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        """
        Generic search:
          - kind: filter by artifact.kind
          - labels: exact-match filter on labels[k] == v
          - metric: if provided with mode, order by metrics[metric]
          - mode: "max" or "min" for metric
          - limit: optional max number of results
        """
        ...

    async def best(
        self,
        *,
        kind: str,
        metric: str,
        mode: Literal["max", "min"],
        filters: dict[str, Any] | None = None,
    ) -> Artifact | None:
        """Return the single best artifact for metric under optional label filters."""
        ...

    async def pin(self, artifact_id: str, pinned: bool = True) -> None:
        """Mark/unmark an artifact as pinned."""
        ...

    async def record_occurrence(
        self,
        a: Artifact,
        extra_labels: dict | None = None,
    ) -> None:
        """
        Append-only lineage log: "this artifact was used/created here".
        """
        ...
