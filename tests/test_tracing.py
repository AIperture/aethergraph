from __future__ import annotations

from aethergraph.services.tracing import summarize_payload


def test_trace_payload_summary_hashes_and_preview() -> None:
    summary = summarize_payload({"prompt": "x" * 400, "count": 3})

    assert summary["metadata"]["type"] == "dict"
    assert summary["metadata"]["count"] == 2
    assert "sha256" in summary["hashes"]
    assert summary["preview"]["prompt"].endswith("...")
