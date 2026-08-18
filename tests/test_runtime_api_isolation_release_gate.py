from pathlib import Path

import aethergraph.api.v1.observability as observability_api
import aethergraph.runtime as runtime_api

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "aethergraph"


def test_public_runtime_does_not_export_internal_composition() -> None:
    forbidden = {"build_default_container", "DefaultContainer"}

    assert forbidden.isdisjoint(runtime_api.__all__)
    assert all(not hasattr(runtime_api, name) for name in forbidden)


def test_superseded_observability_modules_and_router_are_absent() -> None:
    assert not (PACKAGE_ROOT / "observability" / "studio_translation.py").exists()
    assert not (PACKAGE_ROOT / "api" / "v1" / "schemas" / "inspect.py").exists()
    assert not hasattr(observability_api, "trace_router")


def test_observability_contracts_live_below_the_http_api() -> None:
    contracts = PACKAGE_ROOT / "observability" / "contracts.py"
    presenter = PACKAGE_ROOT / "observability" / "inspection.py"

    assert contracts.is_file()
    assert presenter.is_file()
    assert "StudioTranslationPresenter" not in presenter.read_text(encoding="utf-8")
