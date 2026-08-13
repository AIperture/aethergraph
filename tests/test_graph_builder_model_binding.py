from __future__ import annotations

import pytest

from aethergraph.plugins.agents.graph_builder.branches_v2 import _resolve_codegen_llm


class _Context:
    def __init__(self, *, coding: object | None = None) -> None:
        self.coding = coding
        self.calls: list[str | None] = []

    def llm(self, profile: str | None = None) -> object:
        self.calls.append(profile)
        if profile != "coding" or self.coding is None:
            raise KeyError(profile)
        return self.coding


def test_graph_builder_resolves_only_explicit_coding_profile() -> None:
    coding = object()
    context = _Context(coding=coding)

    assert _resolve_codegen_llm(context=context) is coding
    assert context.calls == ["coding"]


def test_graph_builder_does_not_fallback_to_default_profile() -> None:
    context = _Context()

    with pytest.raises(KeyError, match="coding"):
        _resolve_codegen_llm(context=context)

    assert context.calls == ["coding"]
