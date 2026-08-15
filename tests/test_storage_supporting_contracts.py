from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    DocumentQuery,
    DocumentRecord,
    DocumentStore,
    KeyValueQuery,
    KeyValueRecord,
    KeyValueStore,
    PageRequest,
    StorageBundle,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
SCOPE = StorageScope(tenant_id="tenant-1", project_id="project-1")


def test_supporting_records_are_revisioned_scoped_and_deeply_immutable() -> None:
    value = {"roles": ["reader"]}
    kv = KeyValueRecord(
        namespace="auth.grants",
        key="grant-1",
        value=value,
        revision=1,
        scope=SCOPE,
        updated_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    document = DocumentRecord(
        namespace="registry",
        document_id="manifest-1",
        document={"entry": value},
        revision=2,
        scope=SCOPE,
        updated_at=NOW,
        schema_version=1,
    )
    value["roles"].append("writer")

    assert kv.value["roles"] == ("reader",)
    assert document.document["entry"]["roles"] == ("reader",)
    assert "app_id" not in {item.name for item in fields(KeyValueRecord)}
    assert "client_id" not in {item.name for item in fields(DocumentRecord)}


def test_supporting_records_and_queries_fail_closed() -> None:
    with pytest.raises(ValueError, match="after updated_at"):
        KeyValueRecord(
            namespace="auth.grants",
            key="grant-1",
            value={},
            revision=1,
            scope=SCOPE,
            updated_at=NOW,
            expires_at=NOW,
        )
    with pytest.raises(ValueError, match="positive"):
        DocumentRecord(
            namespace="registry",
            document_id="manifest-1",
            document={},
            revision=0,
            scope=SCOPE,
            updated_at=NOW,
        )
    with pytest.raises(ValueError, match="non-empty"):
        KeyValueQuery(scope=SCOPE, namespace="auth.grants", key_prefix="")
    assert (
        DocumentQuery(
            scope=SCOPE,
            namespace="registry",
            page=PageRequest(limit=25),
            metadata={"kind": "agent"},
        ).metadata["kind"]
        == "agent"
    )


def test_bundle_exposes_supporting_and_auth_stores_by_exact_protocol() -> None:
    hints = get_type_hints(StorageBundle)

    assert hints["kv"] is KeyValueStore
    assert hints["documents"] is DocumentStore
    assert hints["auth_grants"] is KeyValueStore
    assert hints["auth_invites"] is KeyValueStore
    assert hints["registry_manifests"] is DocumentStore


def test_supporting_protocol_docstrings_follow_required_section_order() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for protocol in (KeyValueStore, DocumentStore):
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)
