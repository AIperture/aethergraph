# Run And Register

Source of truth:
- `src/aethergraph/runner/__init__.py`
- `src/aethergraph/core/runtime/graph_runner.py`
- `src/aethergraph/__main__.py`
- `src/aethergraph/server/start.py`

Use this after the workflow module is valid and registered.

## Preflight

- The user may not have the AetherGraph repository locally.
- They only need a Python environment where `aethergraph` is installed.
- If `python -c "import aethergraph"` fails, tell them execution and UI serving cannot work yet.

## Local Run

Use `run_async` when the user wants a direct local execution.

```python
from aethergraph.runner import run_async
from workflow import my_workflow

result = await run_async(my_workflow, inputs={"request_text": "summarize this"})
print(result)
```

`await my_workflow(**inputs)` may also work for graph functions, but prefer `run_async` when following a consistent workflow handoff.

## Local AG UI Server

Use `start_server` when the user wants to interact through AG UI from Python.

```python
from aethergraph import start_server

url, handle = start_server(
    workspace="./aethergraph_data",
    load_paths=["./aethergraph_graphs/my-workflow/workflow.py"],
    project_root=".",
    port=0,
    return_handle=True,
)
print(url)
handle.block()
```

Ask for approval before starting a long-running local server process.

## CLI Command For UI

When the final graph file is known, give the user the direct CLI form from `python -m aethergraph serve`.

For a graph file at `./aethergraph_graphs/my-workflow/workflow.py`:

```bash
python -m aethergraph serve \
  --workspace ./aethergraph_data \
  --project-root . \
  --load-path ./aethergraph_graphs/my-workflow/workflow.py \
  --port 8745 \
  --reuse
```

Notes:
- `--load-path` imports the graph file before server startup, so `as_app` registrations appear in UI immediately.
- `--reuse` prints the existing server URL and exits if that workspace is already running.
- `--port 8745` keeps the local UI/server URL stable. Use `0` only if the user wants an auto-picked port.
- The UI URL will be `http://127.0.0.1:8745/ui` unless the user changes host or port.

If the user wants live reload during development:

```bash
python -m aethergraph serve \
  --workspace ./aethergraph_data \
  --project-root . \
  --load-path ./aethergraph_graphs/my-workflow/workflow.py \
  --port 8745 \
  --reload
```

## Optional CLI Registration

If the skill separately registers a graph source by file, use:

```bash
python -m aethergraph register \
  --workspace ./aethergraph_data \
  --source file \
  --path ./aethergraph_graphs/my-workflow/workflow.py
```

Serving with `--load-path` is still the simplest path for getting the app into UI.

## Post-Generation Prompt

After registration succeeds, ask the user:

1. Run locally with `run_async`
2. Start AG UI with `start_server`
3. Cancel

Only start the server after approval.
