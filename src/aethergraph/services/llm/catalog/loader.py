"""Load, validate, digest, and resolve the packaged production model catalog."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import json

from ..registry import get_provider_descriptor, resolve_endpoint_adapter
from .models import ModelCatalog, ModelCatalogEntry


@lru_cache(maxsize=1)
def load_model_catalog() -> ModelCatalog:
    """Load and cross-validate the packaged production model catalog.

    Intro:
        The loader validates the closed schema and verifies every provider and
        endpoint reference against the canonical registry once per process.

    Examples:
        Load the catalog:
            ```python
            catalog = load_model_catalog()
            assert catalog.schema_version == "aethergraph.model-catalog/v1"
            ```

        Inspect catalog entries:
            ```python
            keys = {entry.catalog_key for entry in load_model_catalog().entries}
            ```

    Args:
        None.

    Returns:
        ModelCatalog: Immutable cross-validated production catalog.

    Notes:
        Loading performs no provider network calls. Clear the function cache
        only in tests that replace package resources.
    """

    resource = files(__package__).joinpath("model_catalog.v1.json")
    catalog = ModelCatalog.model_validate_json(resource.read_text(encoding="utf-8"))
    for entry in catalog.entries:
        provider = get_provider_descriptor(entry.provider_id)
        for endpoint_id in entry.endpoint_ids:
            if endpoint_id not in provider.endpoint_ids:
                raise ValueError(
                    f"Catalog endpoint {endpoint_id!r} is not registered for "
                    f"provider {provider.provider_id!r}."
                )
            resolve_endpoint_adapter(
                entry.provider_id,
                entry.operation,
                endpoint_id=endpoint_id,
            )
    return catalog


def catalog_digest(catalog: ModelCatalog | None = None) -> str:
    """Return the canonical SHA-256 digest for catalog identity pinning.

    Intro:
        Digest calculation uses sorted compact JSON over the validated catalog
        and therefore remains independent from source formatting.

    Examples:
        Digest the production catalog:
            ```python
            digest = catalog_digest()
            assert len(digest) == 64
            ```

        Digest an already loaded catalog:
            ```python
            digest = catalog_digest(load_model_catalog())
            ```

    Args:
        catalog: Optional validated catalog; the production catalog is loaded
            when omitted.

    Returns:
        str: Lowercase hexadecimal SHA-256 catalog digest.

    Notes:
        Credentials, environment values, and discovered model lists never
        contribute to this digest.
    """

    value = catalog or load_model_catalog()
    payload = json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def resolve_model_catalog_entry(
    provider_id: str,
    model_id: str,
    operation: str,
    endpoint_id: str,
    *,
    catalog: ModelCatalog | None = None,
) -> ModelCatalogEntry | None:
    """Resolve one exact production capability entry or return unknown.

    Intro:
        Resolution intersects provider, operation, endpoint, and full model
        match. Highest priority wins; an equal-priority ambiguity fails closed.

    Examples:
        Resolve a cataloged model:
            ```python
            entry = resolve_model_catalog_entry(
                "openai", "gpt-5.6", "chat", "openai_responses"
            )
            assert entry is not None
            ```

        Preserve unknown model truth:
            ```python
            entry = resolve_model_catalog_entry(
                "openai", "future-model", "chat", "openai_responses"
            )
            assert entry is None
            ```

    Args:
        provider_id: Exact registered provider identity.
        model_id: Exact configured provider model identity.
        operation: Required model operation.
        endpoint_id: Exact selected endpoint adapter.
        catalog: Optional validated catalog override for tests.

    Returns:
        ModelCatalogEntry | None: Unique highest-priority matching entry, or
        `None` when capability remains unknown.

    Notes:
        Discovered model IDs do not manufacture catalog capability support.
    """

    value = catalog or load_model_catalog()
    matches = sorted(
        (
            entry
            for entry in value.entries
            if entry.provider_id == str(provider_id or "").strip().lower()
            and entry.operation == operation
            and endpoint_id in entry.endpoint_ids
            and entry.matches(model_id)
        ),
        key=lambda item: item.priority,
        reverse=True,
    )
    if not matches:
        return None
    if len(matches) > 1 and matches[0].priority == matches[1].priority:
        raise ValueError(
            "Ambiguous model catalog entries: "
            + ", ".join(
                item.catalog_key for item in matches if item.priority == matches[0].priority
            )
        )
    return matches[0]


__all__ = ["catalog_digest", "load_model_catalog", "resolve_model_catalog_entry"]
