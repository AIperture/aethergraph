from typing import Protocol

from aethergraph.contracts.services.artifacts import Artifact

"""
Artifact index interface for storing and retrieving artifact metadata.
This is a special index used for tracking artifacts generated during runs.

Typical implementations include:
- FileSystemArtifactIndex: File system-based artifact index for durable storage
- DatabaseArtifactIndex: (future) Database-backed artifact index for scalable storage and querying

Note Artifact index is a specialized index for artifacts, different from general document or blob stores. 
"""


class ArtifactIndex(Protocol):
    async def upsert(self, a: Artifact) -> None: ...
    async def list_for_run(self, run_id: str) -> list[Artifact]: ...
    async def search(self, **kwargs) -> list[Artifact]: ...
    async def best(self, **kwargs) -> Artifact | None: ...
    async def pin(self, artifact_id: str, pinned: bool = True) -> None: ...
    async def record_occurrence(
        self, a: Artifact, extra_labels: dict | None = None
    ) -> None: ...  # records an occurrence of the artifact
