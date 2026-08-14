from __future__ import annotations

import pytest

from aethergraph import graphify, tool
from aethergraph.core.execution.forward_scheduler import ForwardScheduler
from aethergraph.core.graph.node_state import NodeStatus
from aethergraph.core.runtime.graph_runner import _build_env


@tool(name="emit_value_for_condition_test", outputs=["value"])
def emit_value_for_condition_test(x: int):
    return {"value": x}


@tool(name="conditional_side_for_condition_test", outputs=["side"])
def conditional_side_for_condition_test(value: int):
    return {"side": value + 1}


@tool(name="downstream_after_side_for_condition_test", outputs=["tail"])
def downstream_after_side_for_condition_test(side: int):
    return {"tail": side * 2}


@graphify(name="condition_graph_test", inputs=["x"], outputs=["value"])
def condition_graph_test(x: int):
    first = emit_value_for_condition_test(x=x)
    conditional_side_for_condition_test(
        value=first.value,
        _condition={"op": "eq", "left": first.value, "right": -1},
    )
    return {"value": first.value}


@graphify(name="condition_graph_downstream_test", inputs=["x"], outputs=["value"])
def condition_graph_downstream_test(x: int):
    first = emit_value_for_condition_test(x=x)
    side = conditional_side_for_condition_test(
        value=first.value,
        _condition={"op": "eq", "left": first.value, "right": -1},
    )
    downstream_after_side_for_condition_test(side=side.side)
    return {"value": first.value}


@graphify(name="condition_graph_truthy_test", inputs=["x"], outputs=["value"])
def condition_graph_truthy_test(x: int):
    first = emit_value_for_condition_test(x=x)
    conditional_side_for_condition_test(
        value=first.value,
        _condition={"op": "truthy", "value": first.value},
    )
    return {"value": first.value}


@graphify(name="condition_graph_and_or_test", inputs=["x"], outputs=["value"])
def condition_graph_and_or_test(x: int):
    first = emit_value_for_condition_test(x=x)
    conditional_side_for_condition_test(
        value=first.value,
        _condition={
            "op": "and",
            "args": [
                {"op": "or", "args": [False, {"op": "eq", "left": first.value, "right": 3}]},
                {"op": "truthy", "value": first.value},
            ],
        },
    )
    return {"value": first.value}


async def _run_graph_in_place(graph, inputs: dict):
    """
    Execute the exact TaskGraph instance in-place, so assertions on `graph.node(...)`
    inspect the real executed state rather than a recovered/rebuilt graph instance.
    """
    env, retry, max_conc = await _build_env(graph, inputs)
    graph._validate_and_bind_inputs(inputs)
    sched = ForwardScheduler(
        graph,
        env,
        retry_policy=retry,
        max_concurrency=max_conc,
        stop_on_first_error=True,
        skip_dep_on_failure=True,
    )
    await sched.run()
    return graph


@pytest.mark.asyncio
async def test_condition_false_skips_node():
    graph = condition_graph_test.build()
    await _run_graph_in_place(graph, {"x": 3})
    side_node = graph.find_by_logic("conditional_side_for_condition_test", first=True)
    assert side_node is not None
    assert graph.node(side_node).state.status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_condition_false_skips_downstream_dependents():
    graph = condition_graph_downstream_test.build()
    await _run_graph_in_place(graph, {"x": 3})

    side_node = graph.find_by_logic("conditional_side_for_condition_test", first=True)
    tail_node = graph.find_by_logic("downstream_after_side_for_condition_test", first=True)
    assert side_node is not None
    assert tail_node is not None
    assert graph.node(side_node).state.status == NodeStatus.SKIPPED
    assert graph.node(tail_node).state.status == NodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_condition_truthy_expression_runs_node():
    graph = condition_graph_truthy_test.build()
    await _run_graph_in_place(graph, {"x": 3})

    side_node = graph.find_by_logic("conditional_side_for_condition_test", first=True)
    assert side_node is not None
    assert graph.node(side_node).state.status == NodeStatus.DONE


@pytest.mark.asyncio
async def test_condition_and_or_expression_runs_node():
    graph = condition_graph_and_or_test.build()
    await _run_graph_in_place(graph, {"x": 3})

    side_node = graph.find_by_logic("conditional_side_for_condition_test", first=True)
    assert side_node is not None
    assert graph.node(side_node).state.status == NodeStatus.DONE
