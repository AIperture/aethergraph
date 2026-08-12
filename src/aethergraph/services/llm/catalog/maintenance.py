"""Adjacent validation and reporting commands for the production model catalog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
import json

from .loader import catalog_digest, load_model_catalog
from .models import ModelCatalog


def validate_catalog(
    catalog: ModelCatalog,
    *,
    today: date | None = None,
) -> tuple[str, ...]:
    """Return deterministic maintenance diagnostics for one valid catalog.

    Intro:
        Schema, provider, endpoint, and positive-evidence validation occurs
        during production loading. This pass reports review-time staleness
        without changing capability truth or model matching.

    Examples:
        Validate the production catalog:
            ```python
            diagnostics = validate_catalog(load_model_catalog())
            ```

        Validate against a pinned review date:
            ```python
            diagnostics = validate_catalog(
                load_model_catalog(), today=date(2026, 8, 12)
            )
            ```

    Args:
        catalog: Validated immutable model catalog.
        today: Optional review date; the current local date is used when omitted.

    Returns:
        tuple[str, ...]: Sorted stable diagnostic strings; empty means no
        maintenance issue was found.

    Notes:
        Stale entries remain resolvable until an explicit catalog revision
        changes their facts. This command never silently disables capability.
    """

    review_date = today or date.today()
    diagnostics = [
        f"stale:{entry.catalog_key}:{entry.stale_after.isoformat()}"
        for entry in catalog.entries
        if entry.stale_after is not None and entry.stale_after < review_date
    ]
    return tuple(sorted(diagnostics))


def catalog_report(catalog: ModelCatalog) -> dict[str, object]:
    """Build one non-secret machine-readable production catalog report.

    Intro:
        The report uses the canonical loader and digest implementation and
        summarizes provider, operation, and entry counts for review tooling.

    Examples:
        Report the production catalog:
            ```python
            report = catalog_report(load_model_catalog())
            assert report["digest"]
            ```

        Inspect provider coverage:
            ```python
            providers = catalog_report(load_model_catalog())["providers"]
            ```

    Args:
        catalog: Validated immutable model catalog.

    Returns:
        dict[str, object]: Stable JSON-compatible catalog report.

    Notes:
        Provider model-list discovery is intentionally not performed here and
        cannot manufacture capability entries.
    """

    return {
        "schema_version": catalog.schema_version,
        "catalog_revision": catalog.catalog_revision,
        "digest": catalog_digest(catalog),
        "entry_count": len(catalog.entries),
        "providers": sorted({entry.provider_id for entry in catalog.entries}),
        "operations": sorted({entry.operation for entry in catalog.entries}),
        "diagnostics": list(validate_catalog(catalog)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate or report the packaged catalog through production APIs.

    Intro:
        The command loads the same catalog used by runtime resolution and emits
        either a short validation line or stable JSON report.

    Examples:
        Validate from Python:
            ```python
            exit_code = main(["validate"])
            ```

        Emit the report from Python:
            ```python
            exit_code = main(["report"])
            ```

    Args:
        argv: Optional command arguments excluding the module name.

    Returns:
        int: Zero when validation succeeds; one when maintenance diagnostics
        require review.

    Notes:
        Use `python -m aethergraph.services.llm.catalog validate` from a shell.
    """

    parser = argparse.ArgumentParser(prog="python -m aethergraph.services.llm.catalog")
    parser.add_argument("command", choices=("validate", "report"))
    args = parser.parse_args(argv)
    catalog = load_model_catalog()
    diagnostics = validate_catalog(catalog)
    if args.command == "report":
        print(json.dumps(catalog_report(catalog), sort_keys=True))
    elif diagnostics:
        print("\n".join(diagnostics))
    else:
        print(f"catalog ok {catalog_digest(catalog)}")
    return 1 if diagnostics else 0


__all__ = ["catalog_report", "main", "validate_catalog"]
