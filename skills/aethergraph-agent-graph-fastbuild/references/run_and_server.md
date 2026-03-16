# Run And Server

Source-of-truth references:
- `src/aethergraph/core/runtime/graph_runner.py`
- `src/aethergraph/server/start.py`

## Run Graphs Quickly

Preferred async entrypoint:

```python
from aethergraph.runner import run_async

result = await run_async(target_graph_or_graph_fn, inputs={"x": 1})
```

Sync convenience:

```python
from aethergraph.runner import run

result = run(target_graph_or_graph_fn, inputs={"x": 1})
```

Notes:
- For `GraphFunction`, `await my_graph_fn(**inputs)` is also valid.
- `run_async` accepts runtime overrides like `run_id`, `session_id`, `agent_id`, `app_id`, `max_concurrency`.

## Start Sidecar Server

```python
from aethergraph import start_server

url = start_server(
    workspace="./aethergraph_workspace",
    port=0,
    load_paths=["./my_graphs.py"],
    project_root=".",
)
print(url)
```

Useful options:
- `return_handle=True` gives `handle.block()` and `handle.stop()`.
- `return_container=True` returns the runtime container.
- `start_server_async(**kw)` exists for async-friendly usage.

Pattern for scripts:

```python
url, handle = start_server(workspace="./aethergraph_workspace", port=0, return_handle=True)
print("Server:", url)
handle.block()
```
