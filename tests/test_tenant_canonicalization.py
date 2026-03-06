from __future__ import annotations

from aethergraph.api.v1.deps import RequestIdentity, _rate_key
from aethergraph.services.registry.unified_registry import UnifiedRegistry


def test_demo_rate_key_uses_resolved_user_id() -> None:
    identity = RequestIdentity(
        user_id="demo:browser-a",
        org_id="demo",
        client_id="browser-a",
        mode="demo",
    )
    assert _rate_key(identity) == "demo:browser-a"


def test_registry_tenant_ignores_client_id_dimension() -> None:
    reg = UnifiedRegistry()
    marker = object()

    reg.register(
        nspace="graph",
        name="demo_graph",
        version="0.1.0",
        obj=marker,
        tenant={"org_id": "o1", "user_id": "u1", "client_id": "client-a"},
    )

    loaded = reg.get_graph(
        "demo_graph",
        tenant={"org_id": "o1", "user_id": "u1", "client_id": "client-b"},
        include_global=True,
    )
    assert loaded is marker
