from typing import Any, Protocol

"""
Document store interface for storing and retrieving JSON-like documents.

Typical implementations include:
- InMemoryDocStore: Transient, in-memory document store for testing or ephemeral use cases
- FileSystemDocStore: File system-based document store for durable storage
- DatabaseDocStore: (future) Database-backed document store for scalable storage and querying

It is used in various parts of the system for storing structured documents.
- memory persistence saving summary JSON documents
- continuation storage for saving intermediate results and tokens
- graph state store for saving state snapshots
"""


class DocStore(Protocol):
    """
    Generic doc_id → JSON dict store.

    NOTE:
      - This is intentionally low-level and schema-agnostic.
      - Higher-level stores (RunStore, EventLog, ArtifactIndex, etc.) are responsible
        for applying domain-specific filtering, sorting, and pagination.
      - DocStore.list() may scan all docs; it is not intended as a scalable
        query interface for large, structured datasets.
    """

    async def put(self, doc_id: str, doc: dict[str, Any]) -> None: ...
    async def get(self, doc_id: str) -> dict[str, Any] | None: ...
    async def delete(self, doc_id: str) -> None: ...

    # Optional
    async def list(self) -> list[str]: ...
