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
    async def put(self, doc_id: str, doc: dict[str, Any]) -> None: ...
    async def get(self, doc_id: str) -> dict[str, Any] | None: ...
    async def delete(self, doc_id: str) -> None: ...

    # Optional
    async def list(self) -> list[str]: ...
