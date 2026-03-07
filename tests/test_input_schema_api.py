from __future__ import annotations

from aethergraph.api.v1.input_schema import merge_input_schema_overrides
from aethergraph.api.v1.schemas.input_schema import InputFieldSpec


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
