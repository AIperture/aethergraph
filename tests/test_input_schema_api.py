from __future__ import annotations

from types import SimpleNamespace

from aethergraph.api.v1.input_schema import merge_input_schema_overrides
from aethergraph.api.v1.schemas.input_schema import InputFieldSpec
from aethergraph.core.graph.io_schema import graph_io_to_slots


def test_graph_io_slots_live_in_the_graph_schema_layer():
    graph = SimpleNamespace(
        spec=SimpleNamespace(
            io=SimpleNamespace(
                required={
                    "message": SimpleNamespace(
                        annotation=str,
                        default=None,
                        required=True,
                        description="User message",
                    )
                },
                optional={
                    "limit": SimpleNamespace(
                        annotation=int,
                        default=5,
                        required=False,
                        description="Maximum results",
                    )
                },
                outputs={
                    "answer": SimpleNamespace(
                        annotation=str,
                        description="Agent answer",
                    )
                },
            )
        ),
        io_signature=lambda **_: {
            "inputs": {"required": ["message"], "optional": {"limit": 5}},
            "outputs": {"keys": ["answer"]},
        },
    )

    slots = graph_io_to_slots(
        graph,
        meta={"io_types": {"inputs": {"message": "string"}}},
    )

    assert [(slot.name, slot.type, slot.required) for slot in slots["inputs"]] == [
        ("message", "string", True),
        ("limit", "number", False),
    ]
    assert slots["inputs"][1].default == 5
    assert [(slot.name, slot.type) for slot in slots["outputs"]] == [("answer", "string")]


def test_merge_input_schema_overrides_accepts_list_shape():
    base = [
        InputFieldSpec(name="tickers", type="string", required=True),
        InputFieldSpec(name="seed", type="string", required=True),
    ]

    merged = merge_input_schema_overrides(
        base,
        app_meta={
            "input_schema": [
                {
                    "name": "tickers",
                    "label": "Tickers",
                    "description": "Comma-separated ticker symbols.",
                    "default": "SPY, AGG, GLD",
                }
            ]
        },
    )

    assert merged[0].name == "tickers"
    assert merged[0].label == "Tickers"
    assert merged[0].description == "Comma-separated ticker symbols."
    assert merged[0].default == "SPY, AGG, GLD"
    assert merged[1].name == "seed"


def test_merge_input_schema_overrides_accepts_dict_shape_for_back_compat():
    base = [
        InputFieldSpec(name="tickers", type="string", required=True),
        InputFieldSpec(name="seed", type="string", required=True),
    ]

    merged = merge_input_schema_overrides(
        base,
        app_meta={
            "input_schema": {
                "tickers": {
                    "label": "Tickers",
                    "description": "Comma-separated ticker symbols.",
                    "default": "SPY, AGG, GLD",
                },
                "seed": {
                    "label": "Seed",
                    "default": "42",
                },
            }
        },
    )

    assert merged[0].label == "Tickers"
    assert merged[0].default == "SPY, AGG, GLD"
    assert merged[1].label == "Seed"
    assert merged[1].default == "42"
